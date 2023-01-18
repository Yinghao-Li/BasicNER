import os
import json
import logging

from typing import Optional
from dataclasses import dataclass, field
from ..base.args import BaseArguments, BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class Arguments(BaseArguments):
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- data arguments ---
    disable_bert_embeddings: Optional[bool] = field(
        default=False, metadata={'help': 'Disable BERT embedding input and use an embedding layer instead.'}
    )

    # --- model arguments ---
    d_hidden: Optional[int] = field(
        default=128, metadata={'help': 'Model hidden dimension.'}
    )
    d_emb: Optional[int] = field(
        default=128, metadata={'help': 'Embedding dimensionality'}
    )


@dataclass
class Config(Arguments, BaseConfig):

    d_emb = None
    vocab = None
    pad_tk_idx = 0
    unk_tk_idx = 1

    def load_vocab(self):
        if self.disable_bert_embeddings:
            assert os.path.exists(os.path.join(self.data_dir, 'vocab.json')), \
                FileNotFoundError("Data folder must have `vocab.json` if not using BERT embeddings.")

            with open(os.path.join(self.data_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)

            self.vocab.insert(self.pad_tk_idx, '<PAD>')
            self.vocab.insert(self.unk_tk_idx, '<UNK>')

        else:
            logger.warning("Using BERT embeddings, will not load pre-defined vocabulary!")

    @property
    def n_vocab(self):
        return len(self.vocab) if self.vocab else 0
