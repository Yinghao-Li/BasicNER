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

    # --- training arguments ---
    warmup_ratio: Optional[int] = field(
        default=0.1, metadata={'help': 'ratio of warmup steps for learning rate scheduler'}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "Default as `linear`. See the documentation of "
                                            "`transformers.SchedulerType` for all possible values"},
    )


@dataclass
class Config(Arguments, BaseConfig):
    def save(self, file_dir: str, file_name: Optional[str] = 'bert-config'):
        BaseConfig.save(self, file_dir, file_name)
        return self

    def load(self, file_dir: str, file_name: Optional[str] = 'bert-config'):
        BaseConfig.load(self, file_dir, file_name)
        return self
