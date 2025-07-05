"""
Datasets.
"""

import torch
from ptls.data_load.augmentations.random_slice import RandomSlice
from ptls.data_load.utils import collate_feature_dict
from torch.utils.data import Dataset


class NextEventDataset(Dataset):

    def __init__(self, df, cat_cols, num_cols,
                 target_cat_cols, target_num_cols,
                 min_length=64, max_length=128,
                 user_col='user_id'):

        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_cat_cols = target_cat_cols
        self.target_num_cols = target_num_cols
        self.min_length = min_length
        self.max_length = max_length
        self.user_col = user_col

        self.seq_cols = cat_cols + num_cols
        df = df[[user_col] + self.seq_cols]

        self.data = df.to_dict(orient='records')

        self.random_slice = RandomSlice(min_len=min_length + 1, max_len=max_length + 1)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]
        data = self.random_slice(data)

        for col in self.target_cat_cols:
            data[f'labels_{col}'] = torch.tensor(data[col][1:])
        for col in self.target_num_cols:
            data[f'labels_{col}'] = torch.tensor(data[col][1:])
        for key in self.seq_cols:
            data[key] = torch.tensor(data[key][:-1])

        return data

    @staticmethod
    def collate_fn(batch):

        padded_batch = collate_feature_dict(batch)

        return padded_batch


class NextEventPedictionDataset(NextEventDataset):

    def __init__(self, df, cat_cols, num_cols,
                 target_cat_cols, target_num_cols,
                 min_length=64, max_length=128,
                 user_col='user_id', validation_mode=False):

        super().__init__(df, cat_cols, num_cols, target_cat_cols, target_num_cols,
                         min_length, max_length, user_col)

        self.validation_mode = validation_mode

    def __getitem__(self, idx):

        data = self.data[idx]
        if self.validation_mode:
            data = self.random_slice(data)
        else:
            for key in self.seq_cols:
                data[key] = data[key][-self.max_length:]

        if self.validation_mode:

            for col in self.target_cat_cols:
                data[f'labels_{col}'] = torch.tensor(data[col][-1])
            for col in self.target_num_cols:
                data[f'labels_{col}'] = torch.tensor(data[col][-1])
            for key in self.seq_cols:
                data[key] = torch.tensor(data[key][:-1])

            return data
        else:

            for key in self.seq_cols:
                data[key] = torch.tensor(data[key])

            return data

    @staticmethod
    def collate_fn(batch):

        padded_batch = collate_feature_dict(batch)

        return padded_batch
