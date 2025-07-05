"""
Lightning module.
"""

import torch
from pytorch_lightning import LightningModule
from torch import nn


class GRUAutoencoderModule(LightningModule):

    def __init__(self, model, lr=1e-3):

        super().__init__()

        self.model = model
        self.lr = lr

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):

        outputs = self.model.forward(batch)
        total_loss, losses = self._compute_loss(batch, outputs)

        self.log("train_loss", total_loss, prog_bar=True)
        self.log('train_loss', total_loss, prog_bar=True)
        for name, loss in losses.items():
            self.log(f'train_{name}', loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        outputs = self.model.forward(batch)
        total_loss, losses = self._compute_loss(batch, outputs)

        self.log('val_loss', total_loss, prog_bar=True)
        for name, loss in losses.items():
            self.log(f'val_{name}', loss, prog_bar=True)

        return total_loss

    def _compute_loss(self, batch, outputs):

        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        losses = {}
        total_loss = 0

        logits, labels = {}, {}
        for key in self.model.vocab_size.keys():
            logits = outputs[key][:, :-1, :]
            labels = batch[key][:, 1:]
            losses[key] = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += losses[key]

        return total_loss, losses

    def predict_step(self, batch, batch_idx):

        outputs = self.model.forward_autoregressive(batch)
        generated = outputs['generated']

        for key in generated.keys():
            generated[key] = generated[key].detach().cpu().numpy()

        return generated


class GRUAutoencoderModuleDayEvent(LightningModule):

    def __init__(self, model, lr=1e-3):

        super().__init__()

        self.model = model
        self.lr = lr

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):

        outputs = self.model.forward(batch)
        total_loss, losses = self._compute_loss(batch, outputs)

        self.log("train_loss", total_loss, prog_bar=True)
        self.log('train_loss', total_loss, prog_bar=True)
        for name, loss in losses.items():
            self.log(f'train_{name}', loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):

        outputs = self.model.forward(batch)
        total_loss, losses = self._compute_loss(batch, outputs)

        self.log('val_loss', total_loss, prog_bar=True)
        for name, loss in losses.items():
            self.log(f'val_{name}', loss, prog_bar=True)

        return total_loss

    def _compute_loss(self, batch, outputs):

        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        losses = {}

        day_logits = outputs['day'][:, :-1, :]
        day_labels = batch['day'][:, 1:]
        event_logits = outputs['event_type'][:, :-1, :]
        event_labels = batch['event_type'][:, 1:]

        losses['day'] = loss_fct(
            day_logits.reshape(-1, day_logits.size(-1)), day_labels.reshape(-1))
        losses['event'] = loss_fct(
            event_logits.reshape(-1, event_logits.size(-1)), event_labels.reshape(-1))
        total_loss = losses['day'] + losses['event']

        return total_loss, losses

    def predict_step(self, batch, batch_idx):

        outputs = self.model.forward_autoregressive(batch)
        generated = outputs['generated']

        for key in generated.keys():
            generated[key] = generated[key].detach().cpu().numpy()

        return generated
