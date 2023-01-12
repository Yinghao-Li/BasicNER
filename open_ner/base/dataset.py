import os
import json
import copy
import regex
import logging
import itertools
import operator
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

import torch
from seqlbtoolkit.data import span_to_label, span_list_to_dict
from seqlbtoolkit.text import split_overlength_bert_input_sequence, substitute_unknown_tokens
from seqlbtoolkit.base_model.dataset import DataInstance


logger = logging.getLogger(__name__)


@dataclass
class BaseDataInstance(DataInstance):
    text: List[str] = None
    lbs: torch.Tensor = None


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 lbs: Optional[List[List[str]]] = None):
        super().__init__()
        self._text = text
        self._lbs = lbs
        self._sent_lens = None
        # Whether the text and lbs sequences are separated according to maximum BERT input lengths
        self._is_separated = False
        self.data_instances = None

    @property
    def n_insts(self):
        return len(self._text)

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    @text.setter
    def text(self, value):
        self._text = value

    @lbs.setter
    def lbs(self, value):
        self._lbs = value

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def __add__(self, other):

        return BaseDataset(
            text=copy.deepcopy(self.text + other.text),
            lbs=copy.deepcopy(self.lbs + other.lbs),
        )

    def __iadd__(self, other):

        self.text = copy.deepcopy(self.text + other.text)
        self.lbs = copy.deepcopy(self.lbs + other.lbs)
        return self

    def prepare(self, config, partition: str):
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
        raise NotImplementedError

    def substitute_unknown_tokens(self, tokenizer_or_name):
        """
        Substitute the tokens in the sequences that cannot be recognized by the tokenizer
        This will not change sequence lengths

        Parameters
        ----------
        tokenizer_or_name: bert tokenizer

        Returns
        -------
        self
        """

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_name) if isinstance(tokenizer_or_name, str) \
            else tokenizer_or_name

        self._text = [substitute_unknown_tokens(tk_seq, tokenizer) for tk_seq in self._text]
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

        if (np.array([len(tk_ids) for tk_ids in tokenizer(
                self._text, add_special_tokens=True, is_split_into_words=True
        ).input_ids]) <= max_seq_length).all():
            self._is_separated = True
            return self

        assert self._sent_lens, AttributeError("To separate sequences, attribute `_sent_lens` cannot be empty!")

        new_text_list = list()
        new_lbs_list = list()

        for sent_lens_inst, text_inst, lbs_inst in zip(self._sent_lens, self._text, self._lbs):

            split_tk_seqs = split_overlength_bert_input_sequence(text_inst, tokenizer, max_seq_length, sent_lens_inst)
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

    def prepare_debug(self):
        for attr in self.__dict__.keys():
            if regex.match(f"^_[a-z]", attr):
                try:
                    setattr(self, attr, getattr(self, attr)[:100])
                except TypeError:
                    pass

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
        sent_tks = data['data']['text']
        tk_seqs.append(sent_tks)

        # get sentence lengths
        if 'sent_lengths' in data['data'].keys():
            sent_lens.append(data['data']['sent_lengths'])
        # get true labels
        lbs = span_to_label(span_list_to_dict(data['label']), sent_tks)
        lbs_list.append(lbs)

    return tk_seqs, lbs_list, sent_lens
