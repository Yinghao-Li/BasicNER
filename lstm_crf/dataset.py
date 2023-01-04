import os
import json
import copy
import regex
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Union, Tuple
from string import printable

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from seqlbtoolkit.text import separate_lengthy_paragraph

from .args import Config
from seqlbtoolkit.data import (
    span_to_label,
    span_list_to_dict,
)

logger = logging.getLogger(__name__)


# noinspection PyBroadException
class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 embs: Optional[List[torch.Tensor]] = None,
                 lbs: Optional[List[List[str]]] = None,
                 ents: Optional[List[str]] = None
                 ):
        super().__init__()
        self._embs = embs
        self._text = text
        self._lbs = lbs
        self._ents = ents

    @property
    def n_insts(self):
        return len(self.obs)

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def embs(self):
        return self._embs if self._embs else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    @property
    def obs(self):
        return self._obs if self._obs else list()

    @property
    def ents(self):
        return self._ents

    @text.setter
    def text(self, value):
        logger.warning(f'{type(self)}: text has been changed')
        self._text = value

    @lbs.setter
    def lbs(self, value):
        logger.warning(f'{type(self)}: labels have been changed')
        self._lbs = value

    @embs.setter
    def embs(self, value):
        logger.warning(f'{type(self)}: embeddings have been changed')
        self._embs = value

    @ents.setter
    def ents(self, value):
        logger.warning(f'{type(self)}: entity types have been changed')
        self._ents = value

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._lbs is not None and len(self._lbs) > 0:
            return self._text[idx], self._embs[idx], self._obs[idx], self._lbs[idx]
        else:
            return self._text[idx], self._embs[idx], self._obs[idx]

    def __add__(self, other: "Dataset") -> "Dataset":
        assert self.ents and other.ents and self.ents == other.ents, ValueError("Entity types not matched!")

        return Dataset(
            text=copy.deepcopy(self.text + other.text),
            embs=copy.deepcopy(self.embs + other.embs),
            lbs=copy.deepcopy(self.lbs + other.lbs),
            ents=copy.deepcopy(self.ents),
        )

    def __iadd__(self, other: "Dataset") -> "Dataset":

        if self.ents:
            assert other.ents and self.ents == other.ents, ValueError("Entity types do not match!")
        else:
            assert other.ents, ValueError("Attribute `ents` not found!")

        self.text = copy.deepcopy(self.text + other.text)
        self.embs = copy.deepcopy(self.embs + other.embs)
        self.lbs = copy.deepcopy(self.lbs + other.lbs)
        self.ents = copy.deepcopy(other.ents)
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

        file_path = os.path.normpath(os.path.join(config.data_dir, ""))
        logger.info(f'Loading data from {file_path}')

        file_dir, file_name = os.path.split(file_path)
        sentence_list, label_list = load_data_from_json(file_path, config.debug_mode)

        # get embedding directory
        emb_name = f"{'.'.join(file_name.split('.')[:-1])}-emb.pt"
        emb_dir = os.path.join(file_dir, emb_name)

        self._text = sentence_list
        self._lbs = label_list
        logger.info(f'Data loaded from {file_path}.')

        logger.info(f'Searching for BERT embeddings...')
        if os.path.isfile(emb_dir):
            logger.info(f"Found embedding file: {emb_dir}. Loading to memory...")
            embs = torch.load(emb_dir)
            if isinstance(embs[0], torch.Tensor):
                self._embs = embs
            elif isinstance(embs[0], np.ndarray):
                self._embs = [torch.from_numpy(emb).to(torch.float) for emb in embs]
            else:
                logger.error(f"Unknown embedding type: {type(embs[0])}")
                raise RuntimeError
        else:
            logger.info(f"{emb_dir} does not exist. Building embeddings instead...")

            self.build_embs(bert_model, config.device, emb_dir)

        self._ents = config.entity_types
        config.d_emb = self._embs[0].shape[-1]
        if getattr(config, 'debug_mode', False):
            self._embs = self._embs[:100]

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
        logger.info(f'Building BERT embeddings with {bert_model} on {device}')
        self._embs = build_embeddings(self._text, bert_model, device)
        if save_dir:
            save_dir = os.path.normpath(save_dir)
            logger.info(f'Saving embeddings to {save_dir}...')
            embs = [emb.numpy().astype(np.float32) for emb in self.embs]
            torch.save(embs, save_dir)
        return self


def batch_prep(emb_list: List[torch.Tensor],
               obs_list: List[torch.Tensor],
               txt_list: Optional[List[List[str]]] = None,
               lbs_list: Optional[List[dict]] = None):
    """
    Pad the instance to the max seq max_seq_length in batch
    All input should already have the dummy element appended to the beginning of the sequence
    """
    for emb, obs, txt, lbs in zip(emb_list, obs_list, txt_list, lbs_list):
        assert len(obs) == len(emb) == len(txt) == len(lbs)
    d_emb = emb_list[0].shape[-1]
    _, n_src, n_obs = obs_list[0].size()
    seq_lens = [len(obs) for obs in obs_list]
    max_seq_len = np.max(seq_lens)

    emb_batch = torch.stack([
        torch.cat([inst, torch.zeros([max_seq_len-len(inst), d_emb])], dim=-2) for inst in emb_list
    ])

    prefix = torch.zeros([1, n_src, n_obs])
    prefix[:, :, 0] = 1
    obs_batch = torch.stack([
        torch.cat([inst, prefix.repeat([max_seq_len-len(inst), 1, 1])]) for inst in obs_list
    ])
    obs_batch /= obs_batch.sum(dim=-1, keepdim=True)

    seq_lens = torch.tensor(seq_lens, dtype=torch.long)

    # we don't need to append the length of txt_list and lbs_list
    return emb_batch, obs_batch, seq_lens, txt_list, lbs_list


def collate_fn(insts):
    """
    Principle used to construct dataloader
    :param insts: original instances
    :return: padded instances
    """
    all_insts = list(zip(*insts))
    if len(all_insts) == 4:
        txt, embs, obs, lbs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt, lbs_list=lbs)
    elif len(all_insts) == 3:
        txt, embs, obs = all_insts
        batch = batch_prep(emb_list=embs, obs_list=obs, txt_list=txt)
    else:
        logger.error('Unsupported number of instances')
        raise ValueError('Unsupported number of instances')
    return batch


def build_bert_emb(sents: List[str],
                   tokenizer,
                   model,
                   device: str):
    sent_emb_list = list()

    for sent_tks in tqdm(sents):

        encs = tokenizer(sent_tks, is_split_into_words=True, add_special_tokens=True, return_offsets_mapping=True)
        input_ids = torch.tensor([encs.input_ids], device=device)
        offsets_mapping = np.array(encs.offset_mapping)

        # calculate BERT last layer embeddings
        with torch.no_grad():
            # get the last hidden state from the BERT model
            last_hidden_states = model(input_ids)[0].squeeze(0).to('cpu')
            # remove the token embeddings regarding the [CLS] and [SEP]
            trunc_hidden_states = last_hidden_states[1:-1, :]

        ori2bert_tk_ids = list()
        idx = 0
        for tk_start in (offsets_mapping[1:-1, 0] == 0):
            if tk_start:
                ori2bert_tk_ids.append([idx])
            else:
                ori2bert_tk_ids[-1].append(idx)
            idx += 1

        emb_list = list()
        for ids in ori2bert_tk_ids:
            embeddings = trunc_hidden_states[ids, :]  # first dim could be 1 or n
            emb_list.append(embeddings.mean(dim=0))

        # add back the embedding of [CLS] as the sentence embedding
        emb_list = [last_hidden_states[0, :]] + emb_list
        bert_emb = torch.stack(emb_list)
        assert not bert_emb.isnan().any(), ValueError('NaN Embeddings!')
        sent_emb_list.append(bert_emb.detach().cpu())

    return sent_emb_list


# noinspection PyTypeChecker
def build_embeddings(src_sents, bert_model, device):

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model).to(device)

    separated_sentences = list()
    ori2sep_ids_map = list()
    n = 0

    # update input sentences so that every sentence has BERT length < 510
    logger.info('Checking lengths. Paragraphs longer than 512 tokens will be separated.')
    for sent_tks in src_sents:
        sent_tks_list, _, _ = separate_lengthy_paragraph(sent_tks, tokenizer)
        n_seps = len(sent_tks_list)
        separated_sentences += sent_tks_list

        ori2sep_ids_map.append(list(range(n, n + n_seps)))
        n += n_seps

    logger.info('Constructing embeddings...')
    sent_emb_list = build_bert_emb(separated_sentences, tokenizer, model, device)

    # Combine embeddings so that the embedding lengths equal to the lengths of the original sentences
    logger.info('Combining results...')
    comb_sent_emb_list = list()

    for sep_ids in ori2sep_ids_map:
        cat_emb = None

        for sep_idx in sep_ids:
            if cat_emb is None:
                cat_emb = sent_emb_list[sep_idx]
            else:
                cat_emb = torch.cat([cat_emb, sent_emb_list[sep_idx][1:]], dim=0)

        assert cat_emb is not None, ValueError('Empty sentence BERT embedding!')
        comb_sent_emb_list.append(cat_emb)

    # The embeddings of [CLS] + original tokens
    for emb, sent_tks in zip(comb_sent_emb_list, src_sents):
        assert len(emb) == len(sent_tks) + 1

    return comb_sent_emb_list


def load_data_from_json(file_dir: str, debug: Optional = None):
    """
    Load data stored in the current data format.


    Parameters
    ----------
    file_dir: file directory
    debug: debugging mode

    """
    with open(file_dir, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    sentence_list = list()
    lbs_list = list()

    for i in range(len(data_dict)):
        data = data_dict[str(i)]
        # get tokens
        tks = [regex.sub("[^{}]+".format(printable), "", tk) for tk in data['data']['text']]
        sent_tks = ['[UNK]' if not tk else tk for tk in tks]
        sentence_list.append(sent_tks)
        # get true labels
        lbs = span_to_label(span_list_to_dict(data['label']), sent_tks)
        lbs_list.append(lbs)

    if debug:
        return sentence_list[:100], lbs_list[:100]
    return sentence_list, lbs_list
