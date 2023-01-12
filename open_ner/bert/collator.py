# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from typing import List
from seqlbtoolkit.base_model.dataset import Batch, instance_list_to_feature_lists
from transformers import DataCollatorForTokenClassification

from .dataset import BertDataInstance


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: List[BertDataInstance], return_tensors=None):

        tk_ids, attn_masks, lbs = instance_list_to_feature_lists(
            instance_list, ['bert_tk_ids', 'bert_attn_masks', 'bert_lbs']
        )
        padded_inputs = self.tokenizer.pad({
            'input_ids': tk_ids, 'attention_mask': attn_masks
        })
        tk_ids = torch.tensor(padded_inputs.input_ids, dtype=torch.int64)
        attn_masks = torch.tensor(padded_inputs.attention_mask, dtype=torch.int64)

        max_len = tk_ids.shape[1]

        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            lbs = torch.stack([
                torch.cat((lb, torch.full((max_len - len(lb), ), self.label_pad_token_id)), dim=0) for lb in lbs
            ])
        else:
            lbs = torch.stack([
                torch.cat((torch.full((max_len - len(lb), ), self.label_pad_token_id), lb), dim=0) for lb in lbs
            ])

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
