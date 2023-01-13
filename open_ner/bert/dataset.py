import os
import logging
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

import torch
from seqlbtoolkit.base_model.dataset import feature_lists_to_instance_list
from tokenizations import get_alignments
from transformers import AutoTokenizer, DataCollatorForTokenClassification

from .args import Config
from ..base.dataset import BaseDataset, BaseDataInstance, load_data_from_json


logger = logging.getLogger(__name__)


@dataclass
class BertDataInstance(BaseDataInstance):

    bert_tk_ids = None
    bert_attn_masks = None
    bert_lbs = None


class Dataset(BaseDataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 lbs: Optional[List[List[str]]] = None):
        super().__init__(text=text, lbs=lbs)

        self._bert_tk_ids = None
        self._bert_attn_masks = None
        self._bert_lbs = None
        self._bert_tk_masks = None

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
        processed_data_path = os.path.join(
            config.data_dir, "processed", "bert", f"{bert_model_name}", f"{partition}.pt"
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

            if config.separate_overlength_sequences:
                self.separate_sequence(config.bert_model_name_or_path, config.max_seq_length)
            self.substitute_unknown_tokens(config.bert_model_name_or_path)

            logger.info("Encoding sequences...")
            self.encode(config.bert_model_name_or_path, {lb: idx for idx, lb in enumerate(config.bio_label_types)})

            logger.info(f"Saving pre-processed dataset to {processed_data_path}.")
            self.save(processed_data_path)

        logger.info(f'Data loaded.')

        if config.debug:
            self.prepare_debug()

        self.data_instances = feature_lists_to_instance_list(
            BertDataInstance,
            bert_tk_ids=self._bert_tk_ids, bert_attn_masks=self._bert_attn_masks, bert_lbs=self._bert_lbs
        )
        return self

    def encode(self, tokenizer_name, lb2idx):
        """
        Build BERT token masks as model input

        Parameters
        ----------
        tokenizer_name: the name of the assigned Huggingface tokenizer
        lb2idx: a dictionary that maps the str labels to indices

        Returns
        -------
        self
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        tokenized_text = tokenizer(self._text, add_special_tokens=True, is_split_into_words=True)

        # exclude over-length instances
        encoded_token_lens = np.array([len(tks) for tks in tokenized_text.input_ids])
        assert (encoded_token_lens <= tokenizer.max_model_input_sizes.get(tokenizer_name, 512)).all(), \
            ValueError("One or more sequences are longer than the maximum model input size. "
                       "Consider using `self.separate_sequence` to break them into smaller pieces.")

        self._bert_tk_ids = tokenized_text.input_ids
        self._bert_attn_masks = tokenized_text.attention_mask

        bert_lbs_list = list()
        bert_tk_masks = list()
        for tks, bert_tk_idx_list, lbs in zip(self._text, self._bert_tk_ids, self._lbs):
            tk_mapping, _ = get_alignments(tks, tokenizer.convert_ids_to_tokens(bert_tk_idx_list))

            for tk, map_ids in zip(tks, tk_mapping):
                assert map_ids, ValueError(f"Cannot map token {tk} to BERT tokens!")
            involved_bert_tk_ids = [map_ids[0] for map_ids in tk_mapping]

            bert_lbs = torch.full((len(bert_tk_idx_list), ), -100)
            bert_lbs[involved_bert_tk_ids] = torch.tensor([lb2idx[lb] for lb in lbs])
            bert_lbs_list.append(bert_lbs)

            masks = np.zeros(len(bert_tk_idx_list), dtype=bool)
            masks[involved_bert_tk_ids] = True
            bert_tk_masks.append(masks)

        self._bert_lbs = bert_lbs_list
        self._bert_tk_masks = bert_tk_masks

        return self







