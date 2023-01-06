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

        vocab_size = getattr(config, 'vocab_size', None)
        self.embedding = nn.Embedding(vocab_size, config.d_emb) if vocab_size else None

        rnn = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = rnn(
            config.d_emb,
            config.d_hidden // 2,
            num_layers=num_rnn_layers,
            bidirectional=True,
            batch_first=True
        )
        self.crf = CRF(config.d_emb, config.n_lbs)

    def _build_features(self, sentences):
        masks = sentences.gt(0)
        embeds = self.embedding(sentences.long())

        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]

        return lstm_out, masks

    def loss(self, xs, tags):
        features, masks = self._build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self._build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
