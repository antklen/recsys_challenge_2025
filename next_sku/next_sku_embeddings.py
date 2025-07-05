"""
Get embeddings from next sku prediction.
"""

import os

import hydra
import numpy as np
import pandas as pd
import torch
from ptls.frames import PtlsDataModule
from ptls.nn import TrxEncoder
from ptls.nn.seq_step import LastStepEncoder
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Model
from tqdm.auto import tqdm

from next_sku.dataset import NextEventDataset, NextEventPedictionDataset
from next_sku.model import HuggingfaceEncoder, SeqEncoder
from next_sku.module import NextEventModule
from src.data import UBCData
from src.inference import get_relevant_clients


@hydra.main(version_base=None, config_path="config", config_name="next_sku")
def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if config.submit:
        data_path = config.data_path_raw
    else:
        data_path = os.path.join(config.data_path_raw, 'input')

    ubc_data = UBCData.from_disk(config.data_path_raw, data_path)
    data = prepare_data(ubc_data, drop_short=True)
    train, test = train_test_split(data, test_size=0.02, random_state=42)
    train, validation = train_test_split(train, test_size=0.01, random_state=42)

    data_module = create_datamodule(train, validation, config)
    model = create_model(config)
    trainer, lightning_module = training(model, data_module, config)
    predict(trainer, lightning_module, test, config)

    ubc_data = UBCData.from_disk(config.data_path_raw, data_path)
    inference_data = prepare_data(ubc_data, drop_short=False, relevant=True)
    client_ids, embeddings = inference(model, inference_data, config)

    client_ids, embeddings = get_relevant_clients(client_ids, embeddings, ubc_data.relevant_clients,
                                                  add_missing=config.submit)
    print(embeddings.shape, client_ids.shape)

    if config.submit:
        save_path = os.path.join(config.save_path, 'embeddings_submit', config.name)
    else:
        save_path = os.path.join(config.save_path, 'embeddings_local/', config.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'embeddings.npy'), embeddings.astype('float16'))
    np.save(os.path.join(save_path, 'client_ids.npy'), client_ids)


def prepare_data(ubc_data, drop_short=True, relevant=False):

    ubc_data.add_to_cart['event'] = 1
    ubc_data.remove_from_cart['event'] = 2
    ubc_data.product_buy['event'] = 3

    data = pd.concat([ubc_data.product_buy, ubc_data.add_to_cart, ubc_data.remove_from_cart])
    data = data.sort_values('timestamp')

    print('data shape', data.shape)
    print('unique clients', data.client_id.nunique())
    print('unique sku', data.sku.nunique())
    print('unique categories', data.category.nunique())

    # need at least 2 interactions for training and validation
    if drop_short:
        counts = data.client_id.value_counts()
        clients = counts[counts > 1].index
        data = data[data.client_id.isin(clients)]
        print('after filtering: data shape', data.shape)
        print('after filtering: unique clients', data.client_id.nunique())

    if relevant:
        data = data[data.client_id.isin(ubc_data.relevant_clients)]
        print('after filtering: data shape', data.shape)
        print('after filtering: unique clients', data.client_id.nunique())

    data = data.drop('timestamp', axis=1).groupby('client_id').agg(list).reset_index()
    data.sku = data.sku.map(np.array)
    data.category = data.category.map(np.array)
    data.price = data.price.map(np.array)
    data.event = data.event.map(np.array)

    return data


def create_datamodule(train, validation, config):

    train_dataset = NextEventDataset(train, **config.dataset)
    eval_dataset = NextEventDataset(validation, **config.dataset)

    data_module = PtlsDataModule(
        train_data=train_dataset,
        valid_data=eval_dataset,
        train_batch_size=config.dataloader.batch_size,
        valid_batch_size=config.dataloader.test_batch_size,
        train_num_workers=config.dataloader.num_workers
    )

    return data_module


def create_model(config):

    embeddings = {}
    if 'category' in config.embedding_size:
        embeddings.update(category={'in': 7000, 'out': config.embedding_size.category})
    if 'event' in config.embedding_size:
        embeddings.update(event={'in': 4, 'out': config.embedding_size.event})
    if 'price' in config.embedding_size:
        embeddings.update(price={'in': 100, 'out': config.embedding_size.size})

    numeric = {}
    if 'price' in config.dataset.num_cols:
        numeric.update({'price': 'identity'})

    trx_encoder = TrxEncoder(embeddings=embeddings, numeric_values=numeric)

    gpt_config = {'vocab_size': 2,
                  'n_positions': config.dataset.max_length,
                  'n_embd': trx_encoder.output_size}
    gpt_config.update(config.model_params)

    gpt_encoder = HuggingfaceEncoder(GPT2Model, GPT2Config, gpt_config)
    model = SeqEncoder(trx_encoder, gpt_encoder, tie_weights=True,
                       heads_cat=config.dataset.target_cat_cols,
                       heads_num=config.dataset.target_num_cols,
                       softplus=False)

    return model


def training(model, data_module, config):

    lightning_module = NextEventModule(
        model, user_col=config.dataset.user_col,
        target_cat_cols=config.dataset.target_cat_cols,
        target_num_cols=config.dataset.target_num_cols,
        **config.training_params)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min",
                                   patience=config.patience, verbose=False)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_loss",
                                 mode="min", save_weights_only=True)
    callbacks = [early_stopping, checkpoint]

    trainer = Trainer(callbacks=callbacks, enable_checkpointing=True, **config.trainer_params)
    trainer.fit(model=lightning_module, datamodule=data_module)

    return trainer, lightning_module


def predict(trainer, lightning_module, test, config):

    test_labels = test[[config.dataset.user_col]].copy()
    test_inputs = test[[config.dataset.user_col]].copy()
    for col in config.dataset.cat_cols + config.dataset.num_cols:
        test_inputs[col] = test[col].map(lambda x: x[:-1])
    for col in config.dataset.target_cat_cols + config.dataset.target_num_cols:
        test_labels[col] = test[col].map(lambda x: x[-1])

    predict_dataset = NextEventPedictionDataset(test_inputs, **config.dataset)
    predict_loader = DataLoader(
        predict_dataset, batch_size=config.dataloader.test_batch_size, shuffle=False,
        num_workers=config.dataloader.num_workers, collate_fn=predict_dataset.collate_fn)

    preds = trainer.predict(model=lightning_module, dataloaders=predict_loader)
    preds = combine_preds(preds, user_col=config.dataset.user_col,
                          target_cat_cols=config.dataset.target_cat_cols)
    preds = pd.merge(preds, test_labels, how='inner',
                     on=config.dataset.user_col, suffixes=['', '_true'])

    metrics = compute_metrics(preds, config.dataset.target_cat_cols, config.dataset.target_num_cols)
    print(pd.Series(metrics, name='metrics'))


def combine_preds(preds, user_col='epk_id', target_cat_cols=None):

    user_ids = np.hstack([pred[user_col] for pred in preds])

    data = {}
    for col in preds[0].keys():
        if col == user_col:
            continue
        data[col] = np.vstack([pred[col] for pred in preds]).squeeze()
        if data[col].ndim == 2:
            data[col] = list(data[col])

    result = pd.DataFrame(data, index=user_ids)
    result.index.name = user_col
    result = result.reset_index()

    if target_cat_cols is not None:
        for col in target_cat_cols:
            result[f'{col}_top1'] =  result[col].map(lambda x: x[0])

    return result


def compute_metrics(preds, target_cat_cols, target_num_cols):

    metrics = {}

    for col in target_cat_cols:
        metrics[f'{col} MRR'] = preds.apply(
            lambda x: calc_reciprocal_rank(x, col=col), axis=1).mean()
        metrics[f'{col} HR@1'] = preds.apply(
            lambda x: calc_hit_rate(x, col=col, K=1), axis=1).mean()
        metrics[f'{col} HR@10'] = preds.apply(
            lambda x: calc_hit_rate(x, col=col, K=10), axis=1).mean()

    for col in target_num_cols:
        metrics[f'{col} spearman'] = preds[[f'{col}_true', col]].corr(method='spearman').iloc[0, 1]
        metrics[f'{col} MAE'] = mean_absolute_error(preds[f'{col}_true'], preds[col])
        metrics[f'{col} RMSE'] = np.sqrt(mean_squared_error(preds[f'{col}_true'], preds[col]))
        metrics[f'{col} R2'] = r2_score(preds[f'{col}_true'], preds[col])

    return metrics


def calc_hit_rate(user, col='event_type', K=10):

    if user[f'{col}_true'] in user[f'{col}'][:K]:
        return 1
    else:
        return 0


def calc_reciprocal_rank(user, col='event_type'):

    try:
        return 1/(1 + np.where(user[f'{col}'] == user[f'{col}_true'])[0][0])
    except:
        return 0


def inference(model, inference_data, config):

    model.eval()
    model.to('cuda')

    inference_dataset = NextEventPedictionDataset(inference_data, **config.dataset)
    inference_loader = DataLoader(
        inference_dataset, batch_size=config.dataloader.test_batch_size, shuffle=False,
        num_workers=config.dataloader.num_workers, collate_fn=inference_dataset.collate_fn)

    client_ids, embeddings = [], []
    reducer = LastStepEncoder()
    for batch in tqdm(inference_loader, total=len(inference_loader)):

        client_ids.extend(batch.payload[config.dataset.user_col].cpu().numpy())
        batch = batch.to('cuda')
        with torch.no_grad():
            embeds = model.trx_encoder(batch)
            outputs = model.seq_encoder(embeds)
            outputs = reducer(outputs)
        embeddings.append(outputs.cpu().numpy())

    client_ids = np.array(client_ids)
    embeddings = np.vstack(embeddings)
    print('client_ids shape', client_ids.shape)
    print('embeddings shape', embeddings.shape)

    return client_ids, embeddings


if __name__ == "__main__":

    main()
