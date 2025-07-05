"""
Dataset class.
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class EventSequenceDataset(Dataset):

    def __init__(self, df, vocab_size, max_length=128):

        self.data = df.to_dict(orient='records')
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]

        for key in self.vocab_size.keys():
            if len(data[key]) > self.max_length - 2:
                data[key] = data[key][-self.max_length + 2:]
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
            data[key] = [1] + data[key] + [2]

        return data


class PaddingCollateFn:

    def __init__(self, padding_value=0, labels_padding_value=-100,
                 labels_keys=['labels'], add_mask=True):

        self.padding_value = padding_value
        self.labels_padding_value = labels_padding_value
        self.labels_keys = labels_keys
        self.add_mask = add_mask

    def __call__(self, batch):

        collated_batch = {}

        for key in batch[0].keys():

            if np.isscalar(batch[0][key]):
                collated_batch[key] = torch.tensor([example[key] for example in batch])
                continue

            seq_key = key

            if key in self.labels_keys:
                padding_value = self.labels_padding_value
            else:
                padding_value = self.padding_value

            values = [torch.tensor(example[key]) for example in batch]
            collated_batch[key] = pad_sequence(values, batch_first=True,
                                               padding_value=padding_value)

        if self.add_mask:
            attention_mask = collated_batch[seq_key] != self.padding_value
            collated_batch['attention_mask'] = attention_mask.to(dtype=torch.float32)

        return collated_batch
