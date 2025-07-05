"""
Model class.
"""

import torch
from torch import nn


class GRUAutoencoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim=128,
                 hidden_size=128, num_layers=2, dropout=0.1):

        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        embedding_layers = {}
        for key in self.vocab_size.keys():
            embedding_layers[key] = nn.Embedding(vocab_size[key], embedding_dim, padding_idx=0)
        self.embedding_layers = nn.ModuleDict(embedding_layers)

        self.encoder_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.decoder_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projections
        output_projections = {}
        for key in self.vocab_size.keys():
            output_projections[key] = nn.Linear(hidden_size, vocab_size[key])
        self.output_projections = nn.ModuleDict(output_projections)

    def forward(self, batch, max_length=None):

        if self.training:
            return self.forward_teacher_forcing(batch)
        else:
            return self.forward_autoregressive(batch, max_length)

    def forward_teacher_forcing(self, batch):

        embeddings = self.get_embeddings(batch)
        _, hidden = self.encoder_gru(embeddings)
        encoded_state = self.get_encoded_state(hidden)

        # Decode - using teacher forcing during training
        seq_len = embeddings.size(1)
        encoded_state = encoded_state.unsqueeze(1).expand(-1, seq_len, -1)
        combined = encoded_state + embeddings
        outputs, _ = self.decoder_gru(combined, hidden)

        logits = {}
        for key in self.vocab_size.keys():
            logits[key] = self.output_projections[key](outputs)

        return logits

    def forward_autoregressive(self, batch, max_length=None):
        """Autoregressive decoding for inference."""

        key = list(self.vocab_size.keys())[0]
        batch_size = batch[key].shape[0]
        device = batch[key].device

        # Use provided max_len or original sequence length
        if max_length is None:
            max_length = batch[key].shape[1]

        embeddings = self.get_embeddings(batch)
        _, hidden = self.encoder_gru(embeddings)
        encoded_state = self.get_encoded_state(hidden)
        encoded_state = encoded_state.unsqueeze(1)

        decoder_input, logits, generated = {}, {}, {}
        for key in self.vocab_size.keys():
            # Initialize decoder input with start tokens
            decoder_input[key] = torch.tensor([1], device=device).repeat(batch_size, 1)
            # Initialize outputs
            logits[key] = torch.zeros(batch_size, max_length, self.vocab_size[key]).to(device)
            # Store generated tokens
            generated[key] = [torch.tensor([1], device=batch[key].device).repeat(batch_size, 1)]

        # Autoregressive decoding
        for t in range(max_length):
            # Embed current input
            embeddings = self.get_embeddings(decoder_input)
            combined = encoded_state + embeddings

            # Forward through decoder GRU
            output, hidden = self.decoder_gru(combined, hidden)

            # Predict next tokens
            for key in self.vocab_size.keys():
                logits[key][:, t] = self.output_projections[key](output.squeeze(1))

            # Get predicted tokens (greedy decoding)
            pred = {}
            for key in self.vocab_size.keys(): 
                pred[key] = torch.argmax(logits[key][:, t], dim=1)

            # Store predictions
            for key in self.vocab_size.keys():
                generated[key].append(pred[key].unsqueeze(1))

            # Prepare next input (use predictions)
            decoder_input = {}
            for key in self.vocab_size.keys():
                decoder_input[key] = pred[key].unsqueeze(1)

        for key in self.vocab_size.keys():
            generated[key] = torch.hstack(generated[key])
        generated['client_id'] = batch['client_id']

        result = logits
        result['generated'] = generated

        return result

    def encode(self, batch):

        embeddings = self.get_embeddings(batch)
        _, hidden = self.encoder_gru(embeddings)
        encoded_state = self.get_encoded_state(hidden)

        return encoded_state

    def get_embeddings(self, batch):

        embeddings = []
        for key in self.vocab_size.keys():
            embeddings.append(self.embedding_layers[key](batch[key]))

        # sum embeddings
        embeddings = torch.stack(embeddings)
        embeddings = torch.sum(embeddings, dim=0)

        return embeddings

    def get_encoded_state(self, hidden):

        encoded_state = hidden[-1]
        return encoded_state


class GRUAutoencoderDayEvent(nn.Module):

    def __init__(self, day_vocab_size,  event_vocab_size, embedding_dim=64,
                 hidden_size=128, num_layers=2, dropout=0.1):

        super().__init__()

        self.day_vocab_size = day_vocab_size
        self.event_vocab_size = event_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.day_embedding = nn.Embedding(day_vocab_size, embedding_dim, padding_idx=0)
        self.event_embedding = nn.Embedding(event_vocab_size, embedding_dim, padding_idx=0)

        self.encoder_gru = nn.GRU(
            input_size=2 * embedding_dim,  # Concatenated day and event embeddings
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.decoder_gru = nn.GRU(
            input_size=2 * embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projections
        self.day_output = nn.Linear(hidden_size, day_vocab_size)
        self.event_output = nn.Linear(hidden_size, event_vocab_size)

    def forward(self, batch, max_length=None):

        if self.training:
            return self.forward_teacher_forcing(batch)
        else:
            return self.forward_autoregressive(batch, max_length)

    def forward_teacher_forcing(self, batch):

        embeddings = self.get_embeddings(batch)
        _, hidden = self.encoder_gru(embeddings)
        encoded_state = self.get_encoded_state(hidden, batch)

        # Decode - using teacher forcing during training
        seq_len = embeddings.size(1)
        encoded_state = encoded_state.unsqueeze(1).expand(-1, seq_len, -1)
        combined = encoded_state + embeddings
        outputs, _ = self.decoder_gru(combined, hidden)

        day_logits = self.day_output(outputs)
        event_logits = self.event_output(outputs)

        return {'day': day_logits, 'event_type': event_logits}

    def forward_autoregressive(self, batch, max_length=None):
        """Autoregressive decoding for inference"""

        batch_size = batch['event_type'].shape[0]
        device = batch['event_type'].device

        # Use provided max_len or original sequence length
        if max_length is None:
            max_length = batch['event_type'].shape[1]

        embeddings = self.get_embeddings(batch)
        _, hidden = self.encoder_gru(embeddings)
        encoded_state = self.get_encoded_state(hidden, batch)
        encoded_state = encoded_state.unsqueeze(1)

        # Initialize decoder input with start tokens
        decoder_input = {
            'day': torch.tensor([1], device=device).repeat(batch_size, 1),
            'event_type': torch.tensor([1], device=device).repeat(batch_size, 1)
        }

        # Initialize outputs
        day_logits = torch.zeros(batch_size, max_length, self.day_vocab_size).to(device)
        event_logits = torch.zeros(batch_size, max_length, self.event_vocab_size).to(device)

        # Store generated tokens
        generated_days = [torch.tensor([1], device=batch['day'].device).repeat(batch_size, 1)]
        generated_events = [torch.tensor([1], device=batch['day'].device).repeat(batch_size, 1)]

        # Autoregressive decoding
        for t in range(max_length):
            # Embed current input
            embeddings = self.get_embeddings(decoder_input)
            combined = encoded_state + embeddings

            # Forward through decoder GRU
            output, hidden = self.decoder_gru(combined, hidden)

            # Predict next tokens
            day_logits[:, t] = self.day_output(output.squeeze(1))
            event_logits[:, t] = self.event_output(output.squeeze(1))

            # Get predicted tokens (greedy decoding)
            pred_day = torch.argmax(day_logits[:, t], dim=1)
            pred_event = torch.argmax(event_logits[:, t], dim=1)

            # Store predictions
            generated_days.append(pred_day.unsqueeze(1))
            generated_events.append(pred_event.unsqueeze(1))

            # Prepare next input (use predictions)
            decoder_input = {
                'day': pred_day.unsqueeze(1),
                'event_type': pred_event.unsqueeze(1)
            }

        generated = {'client_id': batch['client_id'],
                     'day': torch.hstack(generated_days),
                     'event_type': torch.hstack(generated_events)}

        return {'day': day_logits, 'event_type': event_logits, 'generated': generated}

    def encode(self, batch):

        embeddings = self.get_embeddings(batch)
        _, hidden = self.encoder_gru(embeddings)
        encoded_state = self.get_encoded_state(hidden, batch)

        return encoded_state

    def get_embeddings(self, batch):

        # Embed both features
        day_emb = self.day_embedding(batch['day'])
        event_emb = self.event_embedding(batch['event_type'])

        # Concatenate embeddings
        embeddings = torch.cat([day_emb, event_emb], dim=2)
        return embeddings

    def get_encoded_state(self, hidden, batch):

        encoded_state = hidden[-1]
        return encoded_state
