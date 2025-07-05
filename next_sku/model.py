"""
Models.
"""

from ptls.data_load.padded_batch import PaddedBatch
from torch import nn


class HuggingfaceEncoder(nn.Module):

    def __init__(self, model_class, config_class, config):

        super().__init__()

        self.transformer_model = model_class(config_class(**config))

    def forward(self, batch):

        inputs_embeds = batch.payload
        attention_mask = batch.seq_len_mask

        transformer_outputs = self.transformer_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        outputs = transformer_outputs.last_hidden_state

        return PaddedBatch(outputs, batch.seq_lens)


class SeqEncoder(nn.Module):

    def __init__(self, trx_encoder, seq_encoder,
                 heads_cat, heads_num,
                 tie_weights=True, padding_idx=0, init_std=0.02,
                 softplus=False, softplus_beta=1):

        super().__init__()

        self.trx_encoder = trx_encoder
        self.seq_encoder = seq_encoder
        self.heads_cat = heads_cat
        self.heads_num = heads_num
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std
        self.softplus = softplus
        self.softplus_beta = softplus_beta

        self.heads = nn.ModuleDict()
        for head_name in heads_cat:
            self.heads[head_name] = nn.Sequential(
                nn.Linear(self.trx_encoder.output_size,
                          self.trx_encoder.embeddings[head_name].weight.size(1)),
                nn.Linear(self.trx_encoder.embeddings[head_name].weight.size(1),
                          self.trx_encoder.embeddings[head_name].weight.size(0), bias=False))
            if self.tie_weights:
                self.heads[head_name][1].weight = self.trx_encoder.embeddings[head_name].weight

        for head_name in heads_num: 
            self.heads[head_name] = nn.Sequential(
                nn.Linear(self.trx_encoder.output_size, self.trx_encoder.output_size),
                nn.Linear(self.trx_encoder.output_size, 1))

        self.init_weights()

    def init_weights(self):

        # initialization in huggingface transformers
        # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L462

        for embeddings in self.trx_encoder.embeddings:
            self.trx_encoder.embeddings[embeddings].weight.data.normal_(mean=0.0, std=self.init_std)

    def forward(self, batch):

        embeds = self.trx_encoder(batch)
        outputs = self.seq_encoder(embeds).payload

        result = {}
        for head_name in self.heads_cat:
            result[head_name] = self.heads[head_name](outputs)
        for head_name in self.heads_num:
            result[head_name] = self.heads[head_name](outputs)
            if self.softplus:
                softplus = nn.Softplus(beta=self.softplus_beta)
                result[head_name] = softplus(result[head_name])

        return result
