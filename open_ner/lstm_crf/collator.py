# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from typing import List
from seqlbtoolkit.base_model.dataset import Batch, instance_list_to_feature_lists

from .dataset import LSTMDataInstance


class DataCollator:
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, instance_list: List[LSTMDataInstance]):

        text_ids, embs, lbs = instance_list_to_feature_lists(instance_list, ['text_ids', 'embs', 'lbs'])

        seq_lengths = [f.size(0) for f in embs] if embs[0] is not None else [len(tk_ids) for tk_ids in text_ids]
        max_length = max(seq_lengths)

        txt_emb_batch = None
        txt_ids_batch = None
        if embs[0] is not None:
            feature_dim = embs[0].size(-1)
            txt_emb_batch = torch.stack([
                torch.cat((tk_embs, torch.zeros(max_length - len(tk_embs), feature_dim)), dim=0) for tk_embs in embs
            ])
        elif text_ids[0] is not None:
            txt_ids_batch = torch.stack([
                torch.tensor(tk_ids + [self.pad_idx] * (max_length - len(tk_ids))) for tk_ids in text_ids
            ])
        else:
            raise ValueError("Text ids and embs cannot be None at the same time!")

        lbs_batch = torch.stack([
            # torch.tensor(lb + [-1] * (max_length - len(lb)), dtype=torch.long) for lb in lbs
            torch.cat((lb, torch.full((max_length - len(lb), ), 0)), dim=0) for lb in lbs
        ])

        padding_mask_batch = torch.arange(max_length)[None, :] < torch.tensor(seq_lengths)[:, None]

        return Batch(input_ids=txt_ids_batch, embs=txt_emb_batch, lbs=lbs_batch, padding_mask=padding_mask_batch)
