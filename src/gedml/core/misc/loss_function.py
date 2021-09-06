import torch
import numpy as np 

from ...config.setting.core_setting import (
    INDICES_TUPLE,
    INDICES_FLAG
)

def get_unique_indices(indices_tuple, labels):
    a, p, _, _ = split_indices(indices_tuple)
    _, unique_idx = np.unique(labels[a].cpu().numpy(), return_index=True)
    return a[unique_idx], p[unique_idx]

def indices_to_pairs(metric_mat, indices_tuple, assert_shape=[2, 3], return_pn=False):
    tuples = indices_tuple[INDICES_TUPLE]
    assert tuples.shape[1] in assert_shape
    
    if tuples.shape[1] == 3:
        pos_pair = metric_mat[tuples[:,0], tuples[:,1]]
        neg_pair = metric_mat[tuples[:,0], tuples[:,2]]
        if return_pn:
            pn_pair = metric_mat[tuples[:,1], tuples[:,2]]
            return pos_pair, neg_pair, pn_pair
        else:
            return pos_pair, neg_pair

    elif tuples.shape[1] == 2:
        flags = indices_tuple[INDICES_FLAG]
        pos_indices = torch.where(flags)[0]
        neg_indices = torch.where(flags^1)[0]
        pos_pair = metric_mat[tuples[pos_indices,0], tuples[pos_indices,1]]
        neg_pair = metric_mat[tuples[neg_indices,0], tuples[neg_indices,1]]
        return pos_pair, neg_pair

    else:
        raise ValueError("Not supported tuple structure!")

def split_indices(indices_tuple):
    tuples = indices_tuple[INDICES_TUPLE]
    if tuples.shape[1] == 3:
        return tuples[:,0], tuples[:,1], tuples[:,0], tuples[:,2]
    elif tuples.shape[1] == 2:
        flags = indices_tuple[INDICES_FLAG]
        pos_indices = torch.where(flags)[0]
        neg_indices = torch.where(flags^1)[0]
        return tuples[pos_indices,0], tuples[pos_indices,1], tuples[neg_indices,0], tuples[neg_indices,1]
    else:
        raise ValueError("Not supported tuple structure!")

def sumexp(x, keep_mask=None, dim=1):
    x_exp = (
        torch.exp(x) * keep_mask
        if keep_mask is not None
        else torch.exp(x)
    )
    x_sumexp = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_sumexp

def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    x_exp = (
        torch.exp(x) * keep_mask
        if keep_mask is not None
        else torch.exp(x)
    )

    if add_one:
        ones = torch.ones(x_exp.size(dim - 1), dtype=x_exp.dtype, device=x_exp.device).unsqueeze(dim)
        x_exp = torch.cat([x_exp, ones], dim=dim)
    
    x_logsumexp = torch.log(
        torch.sum(x_exp, dim=dim, keepdim=True)
    )
    return x_logsumexp



