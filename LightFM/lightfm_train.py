import os
import hydra
import numpy as np
import pandas as pd
from scipy import sparse
from lightfm import LightFM
from lightfm.data import Dataset
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path='config', config_name='lightfm')
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))

    rel_clients = np.load(os.path.join(config.data_path, 'raw', 'input', 'relevant_clients.npy'))

    data_path = os.path.join(config.data_path, 'raw')
    if not config.submit:
        data_path = os.path.join(data_path, 'input')

    print('Loading data')

    cart = pd.read_parquet(os.path.join(data_path, 'add_to_cart.parquet'))
    buy = pd.read_parquet(os.path.join(data_path, 'product_buy.parquet'))
    remove = pd.read_parquet(os.path.join(data_path, 'remove_from_cart.parquet'))

    buy['action'] = 1
    remove['action'] = -0.5
    cart['action'] = 0.7

    data = pd.concat([buy, remove, cart])
    data = data[data['client_id'].isin(rel_clients)]
    data = data.groupby(by=['client_id', 'sku'])['action'].sum().reset_index()

    ALL_USERS = data['client_id'].unique().tolist()
    ALL_ITEMS = data['sku'].unique().tolist()

    dataset = Dataset()
    dataset.fit_partial(ALL_USERS, ALL_ITEMS) 

    user_mappings = dataset.mapping()[0]
    item_mappings = dataset.mapping()[2]
    inv_user_mappings = {v:k for k, v in user_mappings.items()}
    inv_item_mappings = {v:k for k, v in item_mappings.items()} 

    print('Get item features')
    i_features_emb = get_item_features(item_mappings, ALL_ITEMS, config)
    print('Get user features')
    u_features_emb = get_user_features(data, user_mappings, ALL_USERS, data_path)

    train_interactions, train_weights = dataset.build_interactions(data[['client_id', 'sku', 'action']].values)

    model = LightFM(loss=config.model_params.loss, no_components=config.model_params.no_components)

    print('Train model')
    model.fit(train_interactions, user_features=u_features_emb, item_features=i_features_emb,
              sample_weight=train_weights, epochs=config.model_params.epoch, num_threads=config.model_params.num_threads)
    
    print("Get user's embeddings")
    client_ids, embeddings = get_relevant_embeddings(inv_user_mappings, model, config)
    
    if config.submit:
        save_path = os.path.join(config.data_path, 'embeddings_submit')
    else:
        save_path = os.path.join(config.data_path, 'embeddings_local')

    save_path = os.path.join(save_path, config.save_dir_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, 'embeddings.npy'), client_ids.astype('float16'))
    np.save(os.path.join(save_path, 'client_ids.npy'), embeddings.astype('int'))

    print(f'Save embeddings on {save_path}')


def get_item_features(item_mappings, all_items, config):
    items = pd.read_parquet(os.path.join(config.data_path, 'raw', 'product_properties.parquet'))
    items = items[items.sku.isin(all_items)]
    items = items.reset_index().drop(columns='index')
    items.name = items.name.apply(lambda x: x[1:-1].split())
    items.name = items.name.apply(lambda x: np.array(list(map(int, x))))
    items = items.drop(columns=['category', 'price'])
    items.sku = items.sku.map(item_mappings)
    items = items.sort_values(by='sku')

    return sparse.csr_matrix(np.array(items['name'].values.tolist()))
    

def get_user_features(data, user_mappings, all_users, data_path):
    search = pd.read_parquet(os.path.join(data_path, 'search_query.parquet'))
    search = search[search.client_id.isin(all_users)]
    search['query'] = search['query'].apply(lambda x: x[1:-1].split())
    search['query']= search['query'].apply(lambda x: np.array(list(map(int, x))))
    search = search.groupby('client_id')['query'].mean().reset_index()
    search.client_id = search.client_id.map(user_mappings)
    search = pd.DataFrame({'client_id': data.client_id.map(user_mappings).unique()}).merge(search, how='left', on='client_id')
    search['query'] = search['query'].apply(lambda x: np.array([-1]*16) if x is np.nan else x)
    search = search.sort_values(by='client_id')

    return sparse.csr_matrix(np.array(search['query'].values.tolist()))
    

def get_relevant_embeddings(inv_user_mappings, model, config):
    rel_clients = np.load(os.path.join(config.data_path, 'raw', 'input', 'relevant_clients.npy'))

    result = pd.DataFrame(model.get_user_representations()[1])
    result['client_id'] = range(result.shape[0])
    result.client_id = result.client_id.map(inv_user_mappings)
    result = result[result.client_id.isin(rel_clients)]

    return result['client_id'].values, result.drop(columns=['client_id']).values
    

if __name__ == '__main__':
    main()
