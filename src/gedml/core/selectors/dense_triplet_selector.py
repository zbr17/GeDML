import numpy as np
import torch

from .base_selector import BaseSelector
from ...config.setting.core_setting import (
    INDICES_TUPLE,
    INDICES_FLAG
)

def get_all_triplets_indices(row_labels, col_labels=None, is_same_source=False):
    if col_labels is None:
        col_labels = row_labels.t()
    matches = (row_labels == col_labels).byte()
    diffs = matches ^ 1
    if is_same_source:
        matches.fill_diagonal_(0)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.stack(torch.where(triplets), dim=1)


class DenseTripletSelector(BaseSelector):   
    """
    Select all triplets.
    """ 
    def forward(
        self, 
        metric_mat, 
        row_labels, 
        col_labels, 
        is_same_source=False
    ) -> tuple:
        """
        Select all triplets.
        """
        weight = None
        tuples = get_all_triplets_indices(row_labels, col_labels, is_same_source)
        indices_tuple = {INDICES_TUPLE:tuples, INDICES_FLAG:None}
        return metric_mat, row_labels, col_labels, is_same_source, indices_tuple, weight