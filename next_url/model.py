"""
Model.
"""

from torch import nn
from transformers import GPT2Config, GPT2Model


class GPTRec(nn.Module):

    def __init__(self, gpt_config, vocab_size=None, add_head=True,
                 tie_weights=True, padding_idx=0, init_std=0.02):

        super().__init__()

        self.gpt_config = gpt_config
        self.vocab_size = vocab_size
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=gpt_config['n_embd'],
                                        padding_idx=padding_idx)

        self.transformer_model = GPT2Model(GPT2Config(**gpt_config))

        if self.add_head:
            self.head = nn.Linear(gpt_config['n_embd'], self.vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        # initialization in huggingface transformers
        # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L462

        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()

    def forward(self, input_ids, attention_mask):

        embeds = self.embed_layer(input_ids)
        transformer_outputs = self.transformer_model(
            inputs_embeds=embeds, attention_mask=attention_mask)
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs
