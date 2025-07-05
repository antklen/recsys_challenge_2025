"""
Get embeddings from GRU autoencoder for sku quantized text representation.
"""

import os
import re

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
from gru_autoencoder.model import GRUAutoencoder
from gru_autoencoder.module import GRUAutoencoderModule
from src.data import UBCData
from src.inference import get_relevant_clients


@hydra.main(version_base=None, config_path="config", config_name="gru_ae_sku_text")
def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if config.clearml_project is not None:
        clearml_task = Task.init(project_name=config.clearml_project,
                                 task_name=config.clearml_task,
                                 reuse_last_task_id=False)
        clearml_task.connect(OmegaConf.to_container(config, resolve=True))
    else:
        clearml_task = None

    if config.submit:
        data_path = config.data_path_raw
    else:
        data_path = os.path.join(config.data_path_raw, 'input')

    ubc_data = UBCData.from_disk(config.data_path_raw, data_path)
    data, vocab_size = prepare_data(ubc_data)
    train, validation, test = split_data(data)

    train_loader, eval_loader = create_dataloaders(train, validation, vocab_size, config)
    model = create_model(config, vocab_size)
    trainer, lightning_module = training(model, train_loader, eval_loader, config)
    try:
        predict(trainer, lightning_module, test, config, vocab_size, clearml_task)
    except:
        pass

    inference_data = data[data.client_id.isin(ubc_data.relevant_clients)]
    print('relevant interactions', inference_data.shape[0])
    print('relevant clients', inference_data.client_id.nunique())
    client_ids, embeddings = inference(model, inference_data, config, vocab_size)
    client_ids, embeddings = get_relevant_clients(client_ids, embeddings, ubc_data.relevant_clients,
                                                  add_missing=config.submit)
    print('final shape', embeddings.shape, client_ids.shape)

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


def prepare_data(ubc_data):

    data = pd.concat([ubc_data.product_buy, ubc_data.add_to_cart, ubc_data.remove_from_cart])
    data = pd.merge(data, ubc_data.product_properties)
    data = data.sort_values('timestamp')
    print('data shape', data.shape)
    print('unique clients', data.client_id.nunique())
    print('unique sku', data.sku.nunique())

    data = data.drop_duplicates(['client_id', 'sku'], keep='first')
    print('data shape after removing duplicates', data.shape)

    data['text'] = data['name'].map(parse_to_array)
    data['text'] = data['text'].map(lambda x: x.astype(int))
    data['text'] = data['text'].map(lambda x: x + 3)  # + padding + sos + eos

    vocab_size = {'text': data['text'].map(lambda x: x.max()).max() + 1}
    print('vocab_size', vocab_size)

    data = data.groupby('client_id')['text'].agg(lambda x: np.hstack(x)).reset_index()
    print(data.shape)
    print(data.head())

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

    model = GRUAutoencoder(vocab_size=vocab_size, **config.model_params)

    return model


def training(model, train_loader, eval_loader, config):

    lightning_module = GRUAutoencoderModule(model, **config.training_params)

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


def predict(trainer, lightning_module, test, config, vocab_size, clearml_task=None):

    test_dataset = EventSequenceDataset(test, vocab_size, **config.dataset)
    collate_fn = PaddingCollateFn()
    test_loader = DataLoader(
        test_dataset, batch_size=config.dataloader.test_batch_size, shuffle=False,
        num_workers=config.dataloader.num_workers, collate_fn=collate_fn)

    preds = trainer.predict(model=lightning_module, dataloaders=test_loader)
    preds = combine_preds(preds)
    preds = pd.merge(preds, test, on='client_id', suffixes=['', '_true'])

    cols = ['client_id']
    for col in vocab_size.keys():
        cols.extend([col, col + '_true'])
    preds = preds[cols]
    print('preds shape', preds.shape)
    preds_sample = preds.sample(30).drop('client_id', axis=1).reset_index(drop=True)
    print(preds_sample)

    if clearml_task is not None:
        clearml_logger = clearml_task.get_logger()
        for col in preds_sample.columns:
            if isinstance(preds_sample[col].iloc[0], np.ndarray):
                preds_sample[col] = preds_sample[col].map(list)
        clearml_logger.report_table(title='preds_sample',  series="dataframe",
                                    table_plot=preds_sample)
        clearml_task.upload_artifact('preds_sample', preds_sample)



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


def raise_err_if_incorrect_form(string_representation_of_vector: str):
    """
    Checks if string_representation_of_vector has the correct form.

    Correct form is a string representing list of ints with arbitrary number of spaces in between.

    Args:
        string_representation_of_vector (str): potential string representation of vector
    """
    m = re.fullmatch(r"\[( *\d* *)*\]", string=string_representation_of_vector)
    if m is None:
        raise ValueError(
            f"{string_representation_of_vector} is incorrect form of string representation of vector – correct form is: '[( *\d* *)*]'"
        )


def parse_to_array(string_representation_of_vector: str) -> np.ndarray:
    """
    Parses string representing vector of integers into array of integers.

    Args:
        string_representation_of_vector (str): string representing vector of ints e.g. '[11 2 3]'
    Returns:
        np.ndarray: array of integers obtained from string representation
    """
    raise_err_if_incorrect_form(
        string_representation_of_vector=string_representation_of_vector
    )
    string_representation_of_vector = string_representation_of_vector.replace(
        "[", ""
    ).replace("]", "")
    return np.array(
        [int(s) for s in string_representation_of_vector.split(" ") if s != ""]
    )


if __name__ == "__main__":

    main()
