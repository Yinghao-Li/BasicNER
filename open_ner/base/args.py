import os
import json
import torch
import logging

from typing import Optional
from dataclasses import dataclass, field
from transformers.file_utils import cached_property, torch_required

from seqlbtoolkit.base_model.config import BaseNERConfig
from seqlbtoolkit.data import entity_to_bio_labels

logger = logging.getLogger(__name__)


@dataclass
class BaseArguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- wandb parameters ---
    wandb_api_key: Optional[str] = field(
        default=None, metadata={'help': 'The API key that indicates your wandb account.'
                                        'Can be found here: https://wandb.ai/settings'}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={'help': 'name of the wandb project.'}
    )
    wandb_name: Optional[str] = field(
        default=None, metadata={'help': 'wandb model name.'}
    )

    # --- manage directories and IO ---
    data_dir: Optional[str] = field(
        default='', metadata={'help': 'Directory to datasets'}
    )
    output_dir: Optional[str] = field(
        default='.',
        metadata={"help": "The folder where the models and outputs will be written."},
    )
    bert_model_name_or_path: Optional[str] = field(
        default='', metadata={"help": "Path to pretrained BERT model or model identifier from huggingface.co/models; "
                                      "Used to construct BERT embeddings if not exist"}
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )

    # --- data arguments ---
    separate_overlength_sequences: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether split the overlength sequences into several smaller pieces"
                          "according to their BERT token sequence lengths."}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum length of a BERT token sequence."}
    )
    overwrite_processed_dataset: Optional[bool] = field(
        default=False,
        metadata={'help': "Whether overwrite the processed dataset stored on disk."}
    )
    training_ratio: Optional[float] = field(
        default=None,
        metadata={'help': "Whether down-sampling the training set and the ratio to down-sample"}
    )

    # --- training arguments ---
    batch_size: Optional[int] = field(
        default=16, metadata={'help': 'model training batch size'}
    )
    num_train_epochs: Optional[int] = field(
        default=100, metadata={'help': 'number of denoising model training epochs'}
    )
    learning_rate: Optional[float] = field(
        default=0.001, metadata={'help': 'learning rate of the neural networks in CHMM'}
    )
    weight_decay: Optional[float] = field(
        default=0.01, metadata={'help': 'strength of weight decay'}
    )
    model_buffer_size: Optional[int] = field(
        default=1, metadata={'help': 'How many model checkpoints to buffer for the final evaluation'}
    )
    no_cuda: Optional[bool] = field(
        default=False, metadata={"help": "Disable CUDA even when it is available"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    debug: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )

    def __post_init__(self):
        self.apply_wandb = self.wandb_project and self.wandb_name

    # The following three functions are copied from transformers.training_args
    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda or not torch.cuda.is_available():
            device = torch.device("cpu")
            self._n_gpu = 0
        else:
            device = torch.device("cuda")
            self._n_gpu = 1

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self) -> "int":
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu


@dataclass
class BaseConfig(BaseArguments, BaseNERConfig):

    training_ids = None

    def get_meta(self):

        # Load meta if exist
        meta_dir = os.path.join(self.data_dir, 'meta.json')

        if not os.path.isfile(meta_dir):
            raise FileNotFoundError('Meta file does not exist!')

        with open(meta_dir, 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)

        self.entity_types = meta_dict['entity_types']
        self.bio_label_types = entity_to_bio_labels(meta_dict['entity_types'])

        if self.training_ratio:
            self.training_ids = meta_dict['training_downsampling'][str(self.training_ratio)]

        return self

    @property
    def n_ents(self):
        return len(self.entity_types)

    @property
    def n_lbs(self):
        return len(self.bio_label_types)
