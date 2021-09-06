import numpy as np
import torch

from .base_selector import BaseSelector
from ...config.setting.core_setting import (
    INDICES_TUPLE,
    INDICES_FLAG
)

class DensePairSelector(BaseSelector):  
    """
    Select all pairs.
    """  
    def forward(
        self, 
        metric_mat, 
        row_labels, 
        col_labels, 
        is_same_source=False
    ) -> tuple:
        """
        Select all pairs.
        """
        pos_mask = row_labels == col_labels
        neg_mask = ~pos_mask
        if is_same_source:
            pos_mask.fill_diagonal_(False)
        pos_pairs = torch.stack(torch.where(pos_mask), dim=1)
        neg_pairs = torch.stack(torch.where(neg_mask), dim=1)
        tuples = torch.cat((pos_pairs, neg_pairs), dim=0) 
        pos_flags = torch.ones(pos_pairs.shape[0],1)
        neg_flags = torch.zeros(neg_pairs.shape[0],1)
        flags = torch.cat((pos_flags, neg_flags), dim=0).byte()
        weight = None
        indices_tuple = {INDICES_TUPLE:tuples, INDICES_FLAG:flags}
        return metric_mat, row_labels, col_labels, is_same_source, indices_tuple, weight