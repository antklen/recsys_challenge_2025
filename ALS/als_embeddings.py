import os
import sys

import implicit
import hydra
import numpy as np
import pandas as pd
from implicit.gpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from src.data import UBCData

@hydra.main(version_base=None, config_path='config', config_name='als_embeddings')
def main(config):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)
    data_path = config.data_path_raw
    save_dir = os.path.join(config.save_path, 'embeddings_submit')
    if not config.submit:
        data_path = os.path.join(data_path, 'input')
        save_dir = os.path.join(config.save_path, 'embeddings_local')

    data = UBCData.from_disk(config.data_path_raw, data_path)
    
    train = prepare_page_visits_data(data) if config.item_col == "url" else prepare_buy_add_data(data)
    client_ids, embeddings = get_als_embeddings(train, config.als_factors, item_col = config.item_col)
    client_ids, embeddings = get_relevant_clients(client_ids, embeddings, data.relevant_clients, add_missing=config.submit)
    save_results(save_dir, config.name, client_ids, embeddings)

def get_als_embeddings(data, als_param, item_col, user_col='client_id'):

    data, user_encoder = encode_column(data, col=user_col, new_col='user_id')
    data, item_encoder = encode_column(data, col=item_col, new_col='item_id')  

    csr = csr_matrix((data["weight"], (data["user_id"], data["item_id"])))
    
    model = AlternatingLeastSquares(als_param)
    model.fit(csr)

    client_ids = user_encoder.inverse_transform(range(model.user_factors.shape[0]))
    embeddings = model.user_factors.to_numpy()

    return client_ids, embeddings


def encode_column(df, col, new_col=None, encoder=None):

    if new_col is None:
        new_col = col
    
    if encoder is None:
        encoder = LabelEncoder()
        df[new_col] = encoder.fit_transform(df[col])
        return df, encoder
    else:
        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        df[new_col] = df[col].map(mapping)
        return df


def get_relevant_clients(client_ids, embeddings, relevant_clients, add_missing=False):

    embeddings = embeddings[np.isin(client_ids, relevant_clients)]
    client_ids = client_ids[np.isin(client_ids, relevant_clients)]

    if add_missing:
        relevant_clients_remaining = relevant_clients[~np.isin(relevant_clients, client_ids)]
        client_ids = np.hstack([client_ids, relevant_clients_remaining])
        embeddings = np.vstack([embeddings, np.zeros((len(relevant_clients_remaining), embeddings.shape[1]))])

    return client_ids, embeddings

def prepare_buy_add_data(data):

    data.product_buy['event'] = 'buy'
    data.add_to_cart['event'] = 'add'

    train = pd.concat([data.product_buy, data.add_to_cart])
    train['weight'] = train.event.map({'buy': 3, 'add': 1})
    
    return train

def prepare_page_visits_data(data):

    train = data.page_visit[data.page_visit.client_id.isin(data.relevant_clients)]
    train['weight'] = 1

    return train

def save_results(save_dir, name, client_ids, embeddings):
    save_path = os.path.join(save_dir, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings.astype('float16'))
    np.save(os.path.join(save_path, 'client_ids.npy'), client_ids)

if __name__ == "__main__":
    main()
