"""
Modified from https://github.com/jidasheng/bi-lstm-crf
"""

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF
from .args import Config


class BiRnnCrf(nn.Module):
    def __init__(self, config: Config, num_rnn_layers=1, rnn="lstm"):
        super(BiRnnCrf, self).__init__()

        self._have_embedding_layer = config.disable_bert_embeddings
        self.emb = nn.Embedding(config.n_vocab, config.d_emb, config.pad_tk_idx) if self._have_embedding_layer else None

        rnn = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = rnn(
            config.d_emb,
            config.d_hidden // 2,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        self.crf = CRF(config.d_hidden, config.n_lbs)

    def _build_features(self, inputs):

        embs = self.emb(inputs.input_ids) if self._have_embedding_layer else inputs.embs

        seq_length = inputs.padding_mask.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embs[perm_idx, :]

        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length.cpu(), batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out

    def loss(self, inputs):
        features = self._build_features(inputs)
        loss = self.crf.loss(features, inputs.lbs, inputs.padding_mask)
        return loss

    def forward(self, inputs):
        # Get the emission scores from the BiLSTM
        features = self._build_features(inputs)
        scores, tag_seq = self.crf(features, inputs.padding_mask)
        return scores, tag_seq
