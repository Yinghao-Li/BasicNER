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
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

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
    save_dataset: Optional[bool] = field(
        default=False, metadata={"help": "Whether save the datasets used for training & validation & test"}
    )
    save_dataset_to_data_dir: Optional[bool] = field(
        default=False, metadata={"help": "Whether save the datasets to the original dataset folder. "
                                         "If not, the dataset would be saved to the result folder."}
    )
    load_preprocessed_dataset: Optional[bool] = field(
        default=False, metadata={"help": "Whether load the pre-processed datasets from disk"}
    )
    load_s1_model: Optional[bool] = field(
        default=False, metadata={'help': 'Whether load the trained stage-1 model parameters'}
    )
    load_s2_model: Optional[bool] = field(
        default=False, metadata={'help': 'Whether load the trained stage-2 model parameters.'
                                         'Usually used for testing model.'}
    )
    load_s3_model: Optional[bool] = field(
        default=False, metadata={'help': 'Whether load the trained stage-3 model parameters.'
                                         'Usually used for testing model.'}
    )
    training_ratio_per_epoch: Optional[float] = field(
        default=None, metadata={'help': 'How much data in the training set is used for one epoch.'
                                        'Leave None if use the whole training set'}
    )
    load_init_mat: Optional[bool] = field(
        default=False, metadata={'help': 'Whether to load initial transition and emission matrix from disk'}
    )
    save_init_mat: Optional[bool] = field(
        default=False, metadata={'help': 'Whether to save initial transition and emission matrix from disk'}
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )

    # --- training and data arguments ---
    nn_lr: Optional[float] = field(
        default=0.001, metadata={'help': 'learning rate of the neural networks in CHMM'}
    )
    no_cuda: Optional[bool] = field(
        default=False, metadata={"help": "Disable CUDA even when it is available"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    debug_mode: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )

    def __post_init__(self):
        pass

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
class Config(Arguments, BaseNERConfig):

    def get_meta(self):

        # Load meta if exist
        meta_dir = os.path.join(self.data_dir, 'meta.json')

        if not os.path.isfile(meta_dir):
            raise FileNotFoundError('Meta file does not exist!')

        with open(meta_dir, 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)

        self.entity_types = meta_dict['entity_types']
        self.bio_label_types = entity_to_bio_labels(meta_dict['entity_types'])
