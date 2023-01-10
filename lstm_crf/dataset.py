import os
import json
import copy
import regex
import logging
import itertools
import operator
import numpy as np
from typing import List, Optional
from string import printable
from dataclasses import dataclass
from transformers import AutoTokenizer

import torch
from seqlbtoolkit.embs import build_bert_token_embeddings
from seqlbtoolkit.data import span_to_label, span_list_to_dict
from seqlbtoolkit.text import split_overlength_bert_input_sequence
from seqlbtoolkit.base_model.dataset import (
    DataInstance,
    feature_lists_to_instance_list,
)

from .args import Config


logger = logging.getLogger(__name__)


@dataclass
class NERDataInstance(DataInstance):
    text: List[str] = None
    embs: torch.Tensor = None
    lbs: torch.Tensor = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 embs: Optional[List[torch.Tensor]] = None,
                 lbs: Optional[List[List[str]]] = None):
        super().__init__()
        self._embs = embs
        self._text = text
        self._lbs = lbs
        self._sent_lens = None
        self._data_points = None
        # Whether the text and lbs sequences are separated according to maximum BERT input lengths
        self._is_separated = False

    @property
    def n_insts(self):
        return len(self._text)

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def embs(self):
        return self._embs if self._embs else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    @text.setter
    def text(self, value):
        self._text = value

    @lbs.setter
    def lbs(self, value):
        self._lbs = value

    @embs.setter
    def embs(self, value):
        self._embs = value

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._data_points[idx]

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
        processed_data_path = os.path.join(config.data_dir, "processed", f"{bert_model_name}", f"{partition}.pt")

        if os.path.exists(processed_data_path):

            logger.info(f"Loading pre-processed data from {processed_data_path}.")
            self.load(processed_data_path)

        else:

            logger.info(f'Loading data file: {file_path}')
            sentence_list, label_list, sent_lens = load_data_from_json(file_path)

            self._text = sentence_list
            self._lbs = label_list
            self._sent_lens = sent_lens

            if config.separate_overlength_sequences:
                self.separate_sequence(config.bert_model_name_or_path, config.max_seq_length)

            logger.info("Building BERT embeddings...")
            self.build_embs(config.bert_model_name_or_path, config.device)

            logger.info(f"Saving pre-processed dataset to {processed_data_path}.")
            self.save(processed_data_path)

        if config.debug:
            self._text = self._text[:100]
            self._lbs = self._lbs[:100]
            self._sent_lens = self._sent_lens[:100]
            self._embs = self._embs[:100]

        logger.info(f'Data loaded.')

        config.d_emb = self._embs[0].shape[-1]

        # convert labels to indices
        lb2id_mapping = {lb: idx for idx, lb in enumerate(config.bio_label_types)}
        self._lbs = [torch.tensor([lb2id_mapping[lb] for lb in lbs], dtype=torch.long) for lbs in self._lbs]

        self._data_points = feature_lists_to_instance_list(
            DataInstance,
            text=self._text, embs=self._embs, lbs=self._lbs
        )
        return self

    def separate_sequence(self, tokenizer_or_name, max_seq_length):
        """
        Separate the overlength sequences and separate the labels accordingly

        Parameters
        ----------
        tokenizer_or_name: bert tokenizer
        max_seq_length: maximum bert sequence length

        Returns
        -------
        self
        """
        if self._is_separated:
            logger.warning("The sequences are already separated!")
            return self

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_name) if isinstance(tokenizer_or_name, str) \
            else tokenizer_or_name

        assert self._sent_lens, AttributeError("To separate sequences, attribute `_sent_lens` cannot be empty!")

        new_text_list = list()
        new_lbs_list = list()

        for sent_lens_inst, text_inst, lbs_inst in zip(self._sent_lens, self._text, self._lbs):
            sent_ends = list(itertools.accumulate(sent_lens_inst, operator.add))
            sent_starts = [0] + sent_ends[:-1]
            text = [text_inst[s:e] for s, e in zip(sent_starts, sent_ends)]

            split_tk_seqs = split_overlength_bert_input_sequence(text, tokenizer, max_seq_length)
            split_sq_lens = [len(tk_seq) for tk_seq in split_tk_seqs]

            seq_ends = list(itertools.accumulate(split_sq_lens, operator.add))
            seq_starts = [0] + seq_ends[:-1]

            split_lb_seqs = [lbs_inst[s:e] for s, e in zip(seq_starts, seq_ends)]

            new_text_list += split_tk_seqs
            new_lbs_list += split_lb_seqs

        self._text = new_text_list
        self._lbs = new_lbs_list

        self._is_separated = True

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
                    value = [emb.numpy().astype(np.float32) for emb in value]
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


def load_data_from_json(file_dir: str):
    """
    Load data stored in the current data format.

    Parameters
    ----------
    file_dir: file directory

    """
    with open(file_dir, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    tk_seqs = list()
    sent_lens = list()
    lbs_list = list()

    for i in range(len(data_dict)):
        data = data_dict[str(i)]
        # get tokens
        tks = [regex.sub("[^{}]+".format(printable), "", tk) for tk in data['data']['text']]
        sent_tks = ['[UNK]' if not tk else tk for tk in tks]
        # sent_tks = data['data']['text']
        tk_seqs.append(sent_tks)

        # get sentence lengths
        if 'sent_lengths' in data['data'].keys():
            sent_lens.append(data['data']['sent_lengths'])
        # get true labels
        lbs = span_to_label(span_list_to_dict(data['label']), sent_tks)
        lbs_list.append(lbs)

    return tk_seqs, lbs_list, sent_lens
