"""
Get embeddings from next url prediction.
"""

import os

import hydra
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from replay.metrics import OfflineMetrics, HitRate, MRR, NDCG
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from next_url.dataset import CausalLMDataset, CausalLMPredictionDataset, PaddingCollateFn
from next_url.model import GPTRec
from next_url.module import SeqRecWithSampling
from src.data import UBCData
from src.inference import get_relevant_clients


@hydra.main(version_base=None, config_path="config", config_name="next_url")
def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if config.submit:
        data_path = config.data_path_raw
    else:
        data_path = os.path.join(config.data_path_raw, 'input')

    ubc_data = UBCData.from_disk(config.data_path_raw, data_path)
    data = prepare_page_visits_data(ubc_data, **config.data)

    counts = data.client_id.value_counts()
    clients = counts[counts > 1].index
    data2 = data[data.client_id.isin(clients)]
    print('after filtering short seq: interactions', data2.shape)
    print('after filtering short seq: clients', data2.client_id.nunique())

    train, validation, test = split_data(data2)

    train_loader, eval_loader = create_dataloaders(train, validation, config)
    model = create_model(config, data)
    trainer, seqrec_module = training(model, train_loader, eval_loader, config)
    predict(trainer, seqrec_module, test, config)

    inference_data = data[data.client_id.isin(ubc_data.relevant_clients)]
    print('relevant clients', inference_data.client_id.nunique())
    print('relevant interactions', inference_data.shape)
    client_ids, embeddings = inference(model, inference_data, config)
    client_ids, embeddings = get_relevant_clients(client_ids, embeddings, ubc_data.relevant_clients,
                                                  add_missing=config.submit)
    print(embeddings.shape, client_ids.shape)

    if config.submit:
        save_path = os.path.join(config.save_path, 'embeddings_submit', config.name)
    else:
        save_path = os.path.join(config.save_path, 'embeddings_local', config.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings.astype('float16'))
    np.save(os.path.join(save_path, 'client_ids.npy'), client_ids)


def prepare_page_visits_data(ubc_data, relevant=True, min_count=None):

    print('all clients', ubc_data.page_visit.client_id.nunique())
    print('all url', ubc_data.page_visit.url.nunique())
    print('all interactions', ubc_data.page_visit.shape)

    if relevant:
        data = ubc_data.page_visit[ubc_data.page_visit.client_id.isin(ubc_data.relevant_clients)]
        print('relevant clients', data.client_id.nunique())
        print('relevant url', data.url.nunique())
        print('relevant interactions', data.shape)
    else:
        data = ubc_data.page_visit

    if min_count is not None:
        url_counts = data.url.value_counts()
        urls = url_counts[url_counts >= min_count].index
        data = data[data.url.isin(urls)]
        print('after filtering: clients', data.client_id.nunique())
        print('after filtering: url', data.url.nunique())
        print('after filtering: interactions', data.shape)

    data = data.sort_values('timestamp')
    data = add_time_idx(data, user_col='client_id')

    encoder = LabelEncoder()
    data.url = encoder.fit_transform(data.url)

    return data


def add_time_idx(df, user_col='user_id', timestamp_col='timestamp', sort=True):
    """Add time index to interactions dataframe."""

    if sort:
        df = df.sort_values([user_col, timestamp_col])

    df['time_idx'] = df.groupby(user_col).cumcount()
    df['time_idx_reversed'] = df.groupby(user_col).cumcount(ascending=False)

    return df


def split_data(data):

    client_ids = data.client_id.unique()

    train_clients, test_clients = train_test_split(
        client_ids, test_size=0.02, random_state=42)
    train_clients, validation_clients = train_test_split(
        train_clients, test_size=0.01, random_state=42)

    train = data[data.client_id.isin(train_clients)]
    validation = data[data.client_id.isin(validation_clients)]
    test = data[data.client_id.isin(test_clients)]

    return train, validation, test


def create_dataloaders(train, validation, config):

    val_params = {k: v for k, v in config.dataset.items() if k != 'num_negatives'}
    train_dataset = CausalLMDataset(train, **config.dataset)
    eval_dataset = CausalLMPredictionDataset(validation, validation_mode=True, **val_params)

    collate_fn = PaddingCollateFn()

    train_loader = DataLoader(
        train_dataset, batch_size=config.dataloader.batch_size,
        shuffle=True, num_workers=config.dataloader.num_workers,
        collate_fn=PaddingCollateFn())
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.dataloader.test_batch_size,
        shuffle=False, num_workers=config.dataloader.num_workers,
        collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def create_model(config, data):

    vocab_size = data.url.max() + 1
    gpt_config = {'vocab_size': 2, 'n_positions': config.dataset.max_length}
    gpt_config.update(config.model_params)
    model = GPTRec(gpt_config, vocab_size, add_head=False, tie_weights=True)

    return model


def training(model, train_loader, eval_loader, config):

    seqrec_module = SeqRecWithSampling(model, **config.training_params)

    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=config.patience, verbose=False)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                 mode="max", save_weights_only=True)
    callbacks = [early_stopping, checkpoint]

    trainer = Trainer(callbacks=callbacks, enable_checkpointing=True, **config.trainer_params)
    trainer.fit(model=seqrec_module,
                train_dataloaders=train_loader,
                val_dataloaders=eval_loader)

    return trainer, seqrec_module


def predict(trainer, seqrec_module, test, config):

    test_inputs = test[test.time_idx_reversed > 0]
    test_labels = test[test.time_idx_reversed == 0]
    test_labels = test_labels.rename(columns={config.dataset.user_col: 'user_id',
                                              config.dataset.item_col: 'item_id'})

    params = {k: v for k, v in config.dataset.items() if k != 'num_negatives'}
    predict_dataset = CausalLMPredictionDataset(test_inputs, **params)
    predict_loader = DataLoader(
        predict_dataset, batch_size=config.dataloader.test_batch_size, shuffle=False,
        num_workers=config.dataloader.num_workers, collate_fn=PaddingCollateFn())

    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)
    recs = preds2recs(preds)

    metrics = OfflineMetrics([NDCG(10), MRR(10), HitRate(10)], query_column='user_id',
                             rating_column='prediction')(recs, test_labels)
    print(pd.Series(metrics, name='metrics'))


def preds2recs(preds, item_mapping=None):

    user_ids = np.hstack([pred['user_ids'] for pred in preds])
    scores = np.vstack([pred['scores'] for pred in preds])
    preds = np.vstack([pred['preds'] for pred in preds])

    user_ids = np.repeat(user_ids[:, None], repeats=scores.shape[1], axis=1)

    recs = pd.DataFrame({'user_id': user_ids.flatten(),
                         'item_id': preds.flatten(),
                         'prediction': scores.flatten()})

    if item_mapping is not None:
        recs.item_id = recs.item_id.map(item_mapping)

    return recs


def inference(model, inference_data, config):

    model.eval()
    model.to('cuda')

    params = {k: v for k, v in config.dataset.items() if k != 'num_negatives'}
    inference_dataset = CausalLMPredictionDataset(inference_data, **params)
    inference_loader = DataLoader(
        inference_dataset, batch_size=config.dataloader.test_batch_size, shuffle=False,
        num_workers=config.dataloader.num_workers, collate_fn=PaddingCollateFn())

    client_ids, embeddings = [], []

    for batch in tqdm(inference_loader, total=len(inference_loader)):

        client_ids.extend(batch['user_id'].cpu().numpy())
        for k in batch.keys():
            batch[k] = batch[k].to('cuda')
        with torch.no_grad():
            outputs = model(batch['input_ids'], batch['attention_mask'])
            input_ids = batch['input_ids']
            rows_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
            last_item_idx = (input_ids != 0).sum(axis=1) - 1
            outputs = outputs[rows_ids, last_item_idx, :]
        embeddings.append(outputs.cpu().numpy())

    client_ids = np.array(client_ids)
    embeddings = np.vstack(embeddings)
    print('client_ids shape', client_ids.shape)
    print('embeddings shape', embeddings.shape)

    return client_ids, embeddings


if __name__ == "__main__":

    main()
