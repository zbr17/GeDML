import numpy as np 
import torch

from .base_selector import BaseSelector
from ...config.setting.core_setting import (
    INDICES_TUPLE,
    INDICES_FLAG
)

class RandomTripletSelector(BaseSelector):
    """
    Semi-hard sampling method, euclidean distance metric is required.
    """
    def __init__(self, **kwargs):
        super(BaseSelector, self).__init__(**kwargs)
    
    def forward(
        self,
        metric_mat,
        row_labels,
        col_labels,
        is_same_source=False
    ):
        """
        Randomly select a positive sample and a negative sample
        """
        device = metric_mat.device
        bs = metric_mat.size(0)
        # pos and neg mask
        matches = (row_labels == col_labels).byte()
        diffs = matches ^ 1
        if is_same_source:
            matches.fill_diagonal_(0)

        has_pn_mask = torch.where(
            (torch.sum(matches, dim=-1) > 0) & 
            (torch.sum(diffs, dim=-1) > 0)
        )[0]

        a_ids = torch.arange(bs)[has_pn_mask].to(device)
        # select positive samples
        p_ids = torch.multinomial(
            input=matches.float()[has_pn_mask, :],
            num_samples=1,
            replacement=True
        ).flatten()

        # select negative samples
        n_ids = torch.multinomial(
            input=diffs.float()[has_pn_mask, :],
            num_samples=1,
            replacement=True
        ).flatten()

        indices_tuple = {
            INDICES_TUPLE: torch.stack([a_ids, p_ids, n_ids], dim=1),
            INDICES_FLAG: None
        }
        weight = None
        return (
            metric_mat,
            row_labels,
            col_labels,
            is_same_source,
            indices_tuple,
            weight,
        )