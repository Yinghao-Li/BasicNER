import os
import copy
import regex
import logging
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

import torch
from seqlbtoolkit.embs import build_bert_token_embeddings
from seqlbtoolkit.base_model.dataset import feature_lists_to_instance_list

from .args import Config
from ..base.dataset import BaseDataset, BaseDataInstance, load_data_from_json


logger = logging.getLogger(__name__)


@dataclass
class LSTMDataInstance(BaseDataInstance):
    embs: torch.Tensor = None
    text_ids: List[str] = None


class Dataset(BaseDataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 embs: Optional[List[torch.Tensor]] = None,
                 lbs: Optional[List[List[str]]] = None):
        super().__init__(text=text, lbs=lbs)
        self._embs = embs
        self._text_ids = None

    @property
    def embs(self):
        return self._embs if self._embs else list()

    @embs.setter
    def embs(self, value):
        self._embs = value

    def __add__(self, other: "Dataset") -> "Dataset":

        return Dataset(
            text=copy.deepcopy(self.text + other.text),
            embs=copy.deepcopy(self.embs + other.embs),
            lbs=copy.deepcopy(self.lbs + other.lbs),
        )

    def __iadd__(self, other: "Dataset") -> "Dataset":

        self.text = copy.deepcopy(self.text + other.text)
        self.embs = copy.deepcopy(self.embs + other.embs)
        self.lbs = copy.deepcopy(self.lbs + other.lbs)
        return self

    def prepare(self, config: Config, partition: str):
        """
        Load data from disk

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test]

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        assert partition in ['train', 'valid', 'test'], \
            ValueError(f"Argument `partition` should be one of 'train', 'valid' or 'test'!")

        # Manage paths
        if os.path.isdir(config.bert_model_name_or_path):
            bert_model_name = os.path.normpath(config.bert_model_name_or_path).split(os.sep)[-1]
        else:
            bert_model_name = config.bert_model_name_or_path

        file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.json"))

        if config.disable_bert_embeddings:
            processed_data_path = os.path.join(
                config.data_dir, "processed", "lstm-crf", "no-bert", f"{partition}.pt"
            )
        else:
            processed_data_path = os.path.join(
                config.data_dir, "processed", "lstm-crf", f"{bert_model_name}", f"{partition}.pt"
            )

        if os.path.exists(processed_data_path) and not config.overwrite_processed_dataset:

            logger.info(f"Loading pre-processed data from {processed_data_path}.")
            self.load(processed_data_path)

        else:

            logger.info(f'Loading data file: {file_path}')
            sentence_list, label_list, sent_lens = load_data_from_json(file_path)

            self._text = sentence_list
            self._lbs = label_list
            self._sent_lens = sent_lens

            # use embedding layer
            if config.disable_bert_embeddings:
                tk2idx = {tk: idx for idx, tk in enumerate(config.vocab)}
                self._text_ids = [[tk2idx.get(tk, config.unk_tk_idx) for tk in tk_seq] for tk_seq in self._text]

                self._embs = [None] * len(self._text)

            # use bert embeddings
            else:
                if config.separate_overlength_sequences:
                    self.separate_sequence(config.bert_model_name_or_path, config.max_seq_length)
                self.substitute_unknown_tokens(config.bert_model_name_or_path)

                logger.info("Building BERT embeddings...")
                self.build_embs(config.bert_model_name_or_path, config.device)
                config.d_emb = self._embs[0].shape[-1]

                self._text_ids = [None] * len(self._text)

            logger.info(f"Saving pre-processed dataset to {processed_data_path}.")
            self.save(processed_data_path)

        if config.debug:
            self.prepare_debug()

        if partition == 'train' and config.training_ids:
            self.downsample_training_set(config.training_ids)

        logger.info(f'Data loaded.')

        # convert labels to indices
        lb2id_mapping = {lb: idx for idx, lb in enumerate(config.bio_label_types)}
        self._lbs = [torch.tensor([lb2id_mapping[lb] for lb in lbs], dtype=torch.long) for lbs in self._lbs]

        self.data_instances = feature_lists_to_instance_list(
            LSTMDataInstance,
            text=self._text, text_ids=self._text_ids, embs=self._embs, lbs=self._lbs
        )
        return self

    def build_embs(self,
                   bert_model,
                   device: Optional[torch.device] = torch.device('cpu')) -> "Dataset":
        """
        build bert embeddings

        Parameters
        ----------
        bert_model: the location/name of the bert model to use
        device: device

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        assert bert_model is not None, AssertionError('Please specify BERT model to build embeddings')

        if self._is_separated:
            sent_lens = [[len(tks)] for tks in self._text]
        else:
            sent_lens = self._sent_lens

        logger.info(f'Building BERT embeddings with {bert_model} on {device}')
        self._embs = build_bert_token_embeddings(
            self._text,
            bert_model,
            bert_model,
            device=device,
            sent_lengths_list=sent_lens
        )
        return self

    def save(self, file_path: str):
        """
        Save the entire dataset for future usage

        Parameters
        ----------
        file_path: path to the saved file

        Returns
        -------
        self
        """
        attr_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                if attr == '_embs':
                    try:
                        value = [emb.numpy().astype(np.float32) for emb in value]
                    except AttributeError:
                        pass
                attr_dict[attr] = value

        os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)
        torch.save(attr_dict, file_path)

        return self

    def load(self, file_path: str):
        """
        Load the entire dataset from disk

        Parameters
        ----------
        file_path: path to the saved file

        Returns
        -------
        self
        """
        attr_dict = torch.load(file_path)

        for attr, value in attr_dict.items():
            if attr not in self.__dict__:
                logger.warning(f"Attribute {attr} is not natively defined in dataset!")

            if attr == '_embs':
                value = [torch.from_numpy(emb).to(torch.float) for emb in value]

            setattr(self, attr, value)

        return self
