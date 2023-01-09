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

import torch
from torch.utils.data import DataLoader
from seqlbtoolkit.embs import build_bert_token_embeddings
from seqlbtoolkit.data import span_to_label, span_list_to_dict
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

        file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.json"))
        logger.info(f'Loading data file: {file_path}')

        file_dir, file_name = os.path.split(file_path)
        sentence_list, label_list, sent_lens = load_data_from_json(file_path)

        self._text = sentence_list
        self._lbs = label_list
        self._sent_lens = sent_lens

        if config.debug:
            self._text = self._text[:100]
            self._lbs = self._lbs[:100]
            self._sent_lens = self._sent_lens[:100]

        logger.info(f'Data loaded from {file_path}.')

        logger.info(f'Searching for BERT embeddings...')
        # get embedding directory
        if os.path.isdir(config.bert_model_name_or_path):
            bert_model_name = os.path.normpath(config.bert_model_name_or_path).split(os.sep)[-1]
        else:
            bert_model_name = config.bert_model_name_or_path
        emb_path = os.path.join(file_dir, f"{bert_model_name}", f"{partition}.pt")
        os.makedirs(os.path.join(file_dir, f"{bert_model_name}"), exist_ok=True)

        if os.path.isfile(emb_path):
            logger.info(f"Found embedding file: {emb_path}. Loading to memory...")
            embs = torch.load(emb_path)
            if isinstance(embs[0], torch.Tensor):
                self._embs = embs
            elif isinstance(embs[0], np.ndarray):
                self._embs = [torch.from_numpy(emb).to(torch.float) for emb in embs]
            else:
                logger.error(f"Unknown embedding type: {type(embs[0])}")
                raise RuntimeError
        else:
            logger.info(f"{emb_path} does not exist. Building embeddings instead...")

            self.build_embs(config.bert_model_name_or_path, config.device, emb_path)

        config.d_emb = self._embs[0].shape[-1]

        if config.debug:
            self._embs = self._embs[:100]

        # convert labels to indices
        lb2id_mapping = {lb: idx for idx, lb in enumerate(config.bio_label_types)}
        self._lbs = [torch.tensor([lb2id_mapping[lb] for lb in lbs], dtype=torch.long) for lbs in self._lbs]

        self._data_points = feature_lists_to_instance_list(
            DataInstance,
            text=self._text, embs=self._embs, lbs=self._lbs
        )
        return self

    def build_embs(self,
                   bert_model,
                   device: Optional[torch.device] = torch.device('cpu'),
                   save_dir: Optional[str] = None) -> "Dataset":
        """
        build bert embeddings

        Parameters
        ----------
        bert_model: the location/name of the bert model to use
        device: device
        save_dir: location to update/store the BERT embeddings. Leave None if do not want to save

        Returns
        -------
        self (MultiSrcNERDataset)
        """
        assert bert_model is not None, AssertionError('Please specify BERT model to build embeddings')
        if not self._sent_lens:
            text = self._text
        else:
            sent_ends = [list(itertools.accumulate(sent_lens, operator.add)) for sent_lens in self._sent_lens]
            sent_starts = [[0] + ends[:-1] for ends in sent_ends]
            text = [[text_inst[s:e] for s, e in zip(starts, ends)]
                    for starts, ends, text_inst in zip(sent_starts, sent_ends, self._text)]

        logger.info(f'Building BERT embeddings with {bert_model} on {device}')
        self._embs = build_bert_token_embeddings(text, bert_model, bert_model, device=device)
        if save_dir:
            save_dir = os.path.normpath(save_dir)
            logger.info(f'Saving embeddings to {save_dir}...')
            embs = [emb.numpy().astype(np.float32) for emb in self.embs]
            torch.save(embs, save_dir)
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
        tk_seqs.append(sent_tks)

        # get sentence lengths
        if 'sent_lengths' in data['data'].keys():
            sent_lens.append(data['data']['sent_lengths'])
        # get true labels
        lbs = span_to_label(span_list_to_dict(data['label']), sent_tks)
        lbs_list.append(lbs)

    return tk_seqs, lbs_list, sent_lens
