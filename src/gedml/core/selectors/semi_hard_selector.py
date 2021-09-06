import numpy as np 
import torch

from .base_selector import BaseSelector
from ...config.setting.core_setting import (
    INDICES_TUPLE,
    INDICES_FLAG
)

class SemiHardSelector(BaseSelector):
    """
    Semi-hard sampling method, euclidean distance metric is required.
    """
    def __init__(self, margin=0.2, **kwargs):
        super(BaseSelector, self).__init__(**kwargs)
        self.margin = margin
    
    def forward(
        self,
        metric_mat,
        row_labels,
        col_labels,
        is_same_source=False
    ):
        """
        Randomly select a positive sample and select a negative sample holds:

        :math:`d_p < d_n < d_p + margin`
        """
        device = metric_mat.device
        bs = metric_mat.size(0)
        # pos and neg mask
        matches = (row_labels == col_labels).byte()
        diffs = matches ^ 1
        if is_same_source:
            matches.fill_diagonal_(0)

        has_pos_mask = torch.where(
            torch.sum(matches, dim=-1) > 0
        )[0]

        a_ids = torch.arange(bs)[has_pos_mask].to(device)
        # select positive samples
        p_ids = torch.multinomial(
            input=matches.float()[has_pos_mask, :],
            num_samples=1,
            replacement=True
        ).flatten()
        ap_dist = metric_mat[a_ids, p_ids].unsqueeze(1)

        # select negative samples
        has_pos_metric_mat = metric_mat[has_pos_mask]
        semi_hard_mask = (
            (has_pos_metric_mat > ap_dist) &
            (has_pos_metric_mat < self.margin + ap_dist)
        ).byte() * diffs[has_pos_mask]

        nonzero_semi_hard = torch.where(
            torch.sum(semi_hard_mask, dim=-1) > 0
        )[0] 
        n_ids = torch.multinomial(
            input=semi_hard_mask[nonzero_semi_hard, :].float(),
            num_samples=1,
            replacement=True
        ).flatten()
        a_ids = a_ids[nonzero_semi_hard]
        p_ids = p_ids[nonzero_semi_hard]

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