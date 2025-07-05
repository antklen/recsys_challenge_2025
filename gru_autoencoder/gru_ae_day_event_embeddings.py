"""
Get embeddings from GRU autoencoder for day and event.
"""

import os

import hydra
import numpy as np
import pandas as pd
import torch
from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gru_autoencoder.dataset import EventSequenceDataset, PaddingCollateFn
from gru_autoencoder.model import GRUAutoencoderDayEvent
from gru_autoencoder.module import GRUAutoencoderModuleDayEvent
from src.data import UBCData
from src.inference import get_relevant_clients


@hydra.main(version_base=None, config_path="config", config_name="gru_ae_day_event")
def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if config.clearml_project is not None:
        clearml_task = Task.init(project_name=config.clearml_project,
                                 task_name=config.clearml_task,
                                 reuse_last_task_id=False)
        clearml_task.connect(OmegaConf.to_container(config, resolve=True))

    if config.submit:
        data_path = config.data_path_raw
    else:
        data_path = os.path.join(config.data_path_raw, 'input')

    ubc_data = UBCData.from_disk(config.data_path_raw, data_path)
    data, vocab_size = prepare_data(ubc_data, **config.data)
    train, validation, test = split_data(data)

    train_loader, eval_loader = create_dataloaders(train, validation, vocab_size, config)
    model = create_model(config, vocab_size)
    trainer, lightning_module = training(model, train_loader, eval_loader, config)
    try:
        predict(trainer, lightning_module, test, config, vocab_size)
    except:
        pass

    inference_data = data[data.client_id.isin(ubc_data.relevant_clients)]
    print('relevant interactions', inference_data.shape[0])
    print('relevant clients', inference_data.client_id.nunique())
    client_ids, embeddings = inference(model, inference_data, config, vocab_size)
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

    if config.clearml_project is not None:
        clearml_task.close()


def prepare_data(ubc_data, relevant=True):

    add_to_cart = ubc_data.add_to_cart[['client_id', 'timestamp']]
    remove_from_cart = ubc_data.remove_from_cart[['client_id', 'timestamp']]
    product_buy = ubc_data.product_buy[['client_id', 'timestamp']]
    page_visit = ubc_data.page_visit[['client_id', 'timestamp']]
    search_query = ubc_data.search_query[['client_id', 'timestamp']]

    add_to_cart['event_type'] = 1
    remove_from_cart['event_type'] = 2
    product_buy['event_type'] = 3
    page_visit['event_type'] = 4
    search_query['event_type'] = 5

    data = pd.concat([add_to_cart, remove_from_cart, product_buy, page_visit, search_query])
    print('all data interactions', data.shape[0])
    print('all data unique clients', data.client_id.nunique())

    if relevant:
        data = data[data.client_id.isin(ubc_data.relevant_clients)]
        print('relevant interactions', data.shape[0])
        print('relevant clients', data.client_id.nunique())

    data = data.sort_values('timestamp')

    # prepare categorical features
    date = data.timestamp.dt.date
    delta = date - date.min()
    data['day'] = delta.map(lambda x: x.days) + 3  # + padding + sos + eos tokens
    data.event_type = data.event_type + 2  # + sos + eos tokens

    vocab_size = {
        'event_type': data.event_type.max() + 1,
        'day': data.day.max() + 1
    }

    data = data.drop('timestamp', axis=1).groupby('client_id').agg(list).reset_index()

    return data, vocab_size


def split_data(data):

    train, test = train_test_split(data, test_size=0.02, random_state=42)
    train, validation = train_test_split(train, test_size=0.01, random_state=42)

    return train, validation, test


def create_dataloaders(train, validation, vocab_size, config):

    train_dataset = EventSequenceDataset(train, vocab_size, **config.dataset)
    eval_dataset = EventSequenceDataset(validation, vocab_size, **config.dataset)

    collate_fn = PaddingCollateFn()

    train_loader = DataLoader(
        train_dataset, batch_size=config.dataloader.batch_size,
        shuffle=True, num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn)
    eval_loader = DataLoader(
        eval_dataset, batch_size=config.dataloader.test_batch_size,
        shuffle=False, num_workers=config.dataloader.num_workers,
        collate_fn=collate_fn)

    return train_loader, eval_loader


def create_model(config, vocab_size):

    gru_config = {'day_vocab_size': vocab_size['day'],
                  'event_vocab_size': vocab_size['event_type']}
    gru_config.update(config.model_params)

    model = GRUAutoencoderDayEvent(**gru_config)

    return model


def training(model, train_loader, eval_loader, config):

    lightning_module = GRUAutoencoderModuleDayEvent(model, **config.training_params)

    early_stopping = EarlyStopping(monitor="val_loss", mode="min",
                                   patience=config.patience, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_loss",
                                 mode="min", save_weights_only=True)
    callbacks = [early_stopping, model_summary, checkpoint]

    trainer = Trainer(callbacks=callbacks, enable_checkpointing=True, **config.trainer_params)
    trainer.fit(model=lightning_module,
                train_dataloaders=train_loader,
                val_dataloaders=eval_loader)

    lightning_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

    return trainer, lightning_module


def predict(trainer, lightning_module, test, config, vocab_size):

    test_dataset = EventSequenceDataset(test, vocab_size, **config.dataset)
    collate_fn = PaddingCollateFn()
    test_loader = DataLoader(
        test_dataset, batch_size=config.dataloader.test_batch_size, shuffle=False,
        num_workers=config.dataloader.num_workers, collate_fn=collate_fn)

    preds = trainer.predict(model=lightning_module, dataloaders=test_loader)
    preds = combine_preds(preds)
    preds = pd.merge(preds, test, on='client_id', suffixes=['', '_true'])
    preds = preds[['client_id', 'event_type', 'event_type_true', 'day', 'day_true']]
    print('preds shape', preds.shape)
    print(preds.sample(20).drop('client_id', axis=1).reset_index(drop=True))


def combine_preds(preds, user_col='client_id'):

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

    return result


def inference(model, inference_data, config, vocab_size):

    model.eval()
    model.to('cuda')

    inference_dataset = EventSequenceDataset(inference_data, vocab_size, **config.dataset)
    collate_fn = PaddingCollateFn()
    inference_loader = DataLoader(
        inference_dataset, batch_size=config.dataloader.test_batch_size, shuffle=False,
        num_workers=config.dataloader.num_workers, collate_fn=collate_fn)

    client_ids, embeddings = [], []
    for batch in tqdm(inference_loader, total=len(inference_loader)):

        client_ids.extend(batch['client_id'].cpu().numpy())
        for k in batch.keys():
            batch[k] = batch[k].to('cuda')
        with torch.no_grad():
            encoder_state = model.encode(batch)
        embeddings.append(encoder_state.cpu().numpy())

    client_ids = np.array(client_ids)
    embeddings = np.vstack(embeddings)
    print('client_ids shape', client_ids.shape)
    print('embeddings shape', embeddings.shape)

    return client_ids, embeddings


if __name__ == "__main__":

    main()
