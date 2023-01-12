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

    # --- model arguments ---
    d_hidden: Optional[int] = field(
        default=128, metadata={'help': 'Model hidden dimension.'}
    )


@dataclass
class Config(Arguments, BaseConfig):

    d_emb = None
