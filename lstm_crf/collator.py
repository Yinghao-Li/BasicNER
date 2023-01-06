# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from .dataset import DataInstance
from typing import List


class Batch:
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to(self, device):
        for k, v in self.__dict__.items():
            try:
                setattr(self, k, v.to(device))
            except AttributeError:
                pass
        return self

    def __len__(self):
        for v in list(self.__dict__.values()):
            if isinstance(v, torch.Tensor):
                return v.size(0)


def collator(graphs: List[DataInstance]):

    graphs = [(graph.txt_emb,
               graph.ccp_emb,
               graph.lbs,
               graph.tag_idx,
               graph.sbl_attn_mask,
               graph.pth_attn_mask,
               graph.glb_pos_idx,
               graph.sbl_pos_idx,
               graph.lvl_pos_idx) for graph in graphs]

    txt_emb, ccp_emb, lbs, tag_idx, sbl_attn_mask, pth_attn_mask, glb_pos_idx, sbl_pos_idx, lvl_pos_idx = zip(*graphs)

    n_nodes = [f.size(0) for f in txt_emb]
    n_nodes_max = max(n_nodes)
    feature_dim = txt_emb[0].size(-1)

    txt_emb_batch = torch.stack([
        torch.cat((inst, torch.zeros(n_nodes_max - len(inst), feature_dim)), dim=0) for inst in txt_emb
    ])
    ccp_emb_batch = torch.stack([
        inst.repeat(n_nodes_max, 1) for inst in ccp_emb
    ])
    lbs_batch = torch.stack([
        torch.cat((lb, torch.full((n_nodes_max - len(lb), ), -1)), dim=0) for lb in lbs
    ])
    tag_idx_batch = torch.stack([
        torch.cat((tag, torch.zeros(n_nodes_max - len(tag), dtype=torch.long)), dim=0) for tag in tag_idx
    ])
    sibling_attn_mask_batch = torch.stack([
        F.pad(mask, (0, n_nodes_max-len(mask), 0, n_nodes_max-len(mask)), "constant", False) for mask in sbl_attn_mask
    ])
    parent_attn_mask_batch = torch.stack([
        F.pad(mask, (0, n_nodes_max-len(mask), 0, n_nodes_max-len(mask)), "constant", False) for mask in pth_attn_mask
    ])
    glb_pos_idx_batch = torch.stack([
        torch.cat((inst, torch.zeros(n_nodes_max - len(inst), dtype=torch.long)), dim=0) for inst in glb_pos_idx
    ])
    sbl_pos_idx_batch = torch.stack([
        torch.cat((inst, torch.zeros(n_nodes_max - len(inst), dtype=torch.long)), dim=0) for inst in sbl_pos_idx
    ])
    lvl_pos_idx_batch = torch.stack([
        torch.cat((inst, torch.zeros(n_nodes_max - len(inst), dtype=torch.long)), dim=0) for inst in lvl_pos_idx
    ])

    padding_mask_batch = lbs_batch == -1

    return Batch(
        txt_emb=txt_emb_batch,
        ccp_emb=ccp_emb_batch,
        tag_idx=tag_idx_batch,
        lbs=lbs_batch,
        sbl_attn_mask=sibling_attn_mask_batch,
        pth_attn_mask=parent_attn_mask_batch,
        padding_mask=padding_mask_batch,
        glb_pos_idx=glb_pos_idx_batch,
        sbl_pos_idx=sbl_pos_idx_batch,
        lvl_pos_idx=lvl_pos_idx_batch
    )