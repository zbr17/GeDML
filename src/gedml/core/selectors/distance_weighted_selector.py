import numpy as np 
import torch

from .base_selector import BaseSelector
from ...config.setting.core_setting import (
    INDICES_TUPLE,
    INDICES_FLAG
)

class DistanceWeightedSelector(BaseSelector):
    """
    Distance weighted sampling method, euclidean distance metric is required.

    paper: `Sampling Matters in Deep Embedding Learning <http://openaccess.thecvf.com/content_iccv_2017/html/Wu_Sampling_Matters_in_ICCV_2017_paper.html>`_
    """
    def __init__(self, lower_cutoff=0.5, upper_cutoff=1.4, embedding_dim=512, **kwargs):
        super(DistanceWeightedSelector, self).__init__(**kwargs)
        self.lower_cutoff = lower_cutoff
        self.upper_cutoff = upper_cutoff
        self.embedding_dim = embedding_dim
    
    def forward(
        self,
        metric_mat,
        row_labels,
        col_labels,
        is_same_source=False
    ) -> tuple:
        """
        Randomly select a positive sample for anchor sample and select a negative sample for anchor sample according to the distance weighted probability.

        The distribution of pairwise distances follows:

        :math:`q(d) \propto d^{n-2} [1 - \\frac{1}{4} d^2 ]^{\\frac{n-3}{2}}`
        """
        metric_mat_to_weights = metric_mat.clamp(min=self.lower_cutoff).detach()
        bs = metric_mat.size(0)
        device = metric_mat.device
        # pos and neg mask
        matches = (row_labels == col_labels).byte()
        diffs = matches ^ 1

        # compute distance weighted prob
        minus_log_weights = (
            (2.0 - self.embedding_dim) * torch.log(metric_mat_to_weights) + 
            ((3.0 - self.embedding_dim) / 2.0) * torch.log(1.0 - 0.25 * metric_mat_to_weights.pow(2))
        ) * diffs
        ## avoid inf or nan
        inf_or_nan = torch.isinf(minus_log_weights) | torch.isnan(minus_log_weights)
        ## compute weight
        inv_weights = torch.exp(
            minus_log_weights - torch.max(minus_log_weights[~inf_or_nan])
        )
        ## NOTE: Cutting of values with high distances made the results slightly worse according to Confusezius / Pytorch-Metric-Learning-Baseline
        # inv_weights = (
        #     inv_weights * diffs * (metric_mat_to_weights < self.upper_cutoff).byte()
        # )
        inv_weights[inf_or_nan] = 0

        # construct the tuple
        if is_same_source:
            matches.fill_diagonal_(0)

        # avoid nan, inf and zero
        non_zero_row = torch.where(
            (torch.sum(matches, dim=-1) > 0) & (torch.sum(inv_weights, dim=-1) > 0)
        )[0]
        inv_weights = inv_weights[non_zero_row, :]
        ## anchor indices
        a_ids = torch.arange(bs)[non_zero_row].to(device)
        ## positive indices
        p_ids = torch.multinomial(
            input=matches[non_zero_row, :].float(),
            num_samples=1,
            replacement=True
        ).flatten()
        ## negative indices
        inv_weights = inv_weights / inv_weights.sum(dim=-1, keepdim=True)
        n_ids = torch.multinomial(
            input=inv_weights,
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