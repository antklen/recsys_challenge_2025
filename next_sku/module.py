"""
Pytorch-lightning module.
"""


import pytorch_lightning as pl
import torch
from torch import nn


class NextEventModule(pl.LightningModule):

    def __init__(self, model, target_cat_cols, target_num_cols,
                 loss_coefs, lr=1e-3, padding_idx=0,
                 predict_top_k=10, numeric_loss='mae',
                 user_col='user_id'):

        super().__init__()

        self.model = model
        self.target_cat_cols = target_cat_cols
        self.target_num_cols = target_num_cols
        self.loss_coefs = loss_coefs
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self.numeric_loss = numeric_loss  # mae or rmse
        self.user_col = user_col

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):

        outputs = self.model(batch)
        total_loss, losses = self._compute_loss(outputs, batch)

        batch_size = batch.payload[self.user_col].shape[0]
        self.log('train_loss', total_loss, prog_bar=True, batch_size=batch_size)
        for name, loss in losses.items():
            self.log(f'train_loss_{name}', loss, prog_bar=True, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):

        outputs = self.model(batch)
        total_loss, losses = self._compute_loss(outputs, batch)

        batch_size = batch.payload[self.user_col].shape[0]
        self.log('val_loss', total_loss, prog_bar=True, batch_size=batch_size)
        for name, loss in losses.items():
            self.log(f'val_loss_{name}', loss, prog_bar=True, batch_size=batch_size)

    def _compute_loss(self, outputs, batch):

        losses = {}
        loss_fct_cat = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        if self.numeric_loss == 'mae':
            loss_fct_num = nn.L1Loss(reduction='none')
        elif self.numeric_loss == 'rmse':
            loss_fct_num = nn.MSELoss(reduction='none')

        for col in self.target_cat_cols:
            losses[col] = loss_fct_cat(outputs[col].view(-1, outputs[col].size(-1)),
                                       batch.payload[f'labels_{col}'].view(-1))

        for col in self.target_num_cols:
            losses[col] = loss_fct_num(outputs[col].squeeze(),
                                       batch.payload[f'labels_{col}'].to(torch.float32))
            losses[col] = losses[col].view(-1)[
                batch.payload[self.target_cat_cols[0]].view(-1) != self.padding_idx]
            if self.numeric_loss == 'rmse':
                losses[col] = torch.sqrt(losses[col].mean())
            else:
                losses[col] = losses[col].mean()

        total_loss = 0
        for col in self.target_cat_cols:
            total_loss += self.loss_coefs[col] * losses[col]
        for col in self.target_num_cols:
            total_loss += self.loss_coefs[col] * losses[col]

        return total_loss, losses

    def predict_step(self, batch, batch_idx):

        preds, preds_cat, scores_cat = self._make_prediction(batch)

        result = {}
        result[self.user_col] = batch.payload[self.user_col].detach().cpu().numpy()

        for col, value in preds.items():
            result[col] = value.detach().cpu().numpy()
        for col, value in preds_cat.items():
            result[col] = value.detach().cpu().numpy()
        for col, value in scores_cat.items():
            result[f'scores_{col}'] = value.detach().cpu().numpy()

        return result

    def _make_prediction(self, batch):

        outputs = self.model(batch)

        rows_ids = torch.arange(batch.payload[self.user_col].shape[0], dtype=torch.long,
                                device=batch.payload[self.user_col].device)
        preds = {}
        for col in self.target_cat_cols:
            preds[col] = outputs[col][rows_ids, batch.seq_lens - 1, :]
        for col in self.target_num_cols:
            preds[col] = outputs[col][rows_ids, batch.seq_lens - 1, :]

        scores_cat, preds_cat = {}, {}
        for col in self.target_cat_cols:
            scores_cat[col], preds_cat[col] = torch.sort(preds[col], descending=True)
            scores_cat[col] = scores_cat[col][:, :self.predict_top_k]
            preds_cat[col] = preds_cat[col][:, :self.predict_top_k]

        return preds, preds_cat, scores_cat
