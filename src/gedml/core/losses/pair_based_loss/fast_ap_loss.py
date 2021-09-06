import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class FastAPLoss(BaseLoss):
    """
    modified from: https://github.com/KevinMusgrave/pytorch-metric-learning

    paper: `Deep Metric Learning to Rank <https://openaccess.thecvf.com/content_CVPR_2019/html/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.html>`_
    """
    def __init__(
        self,
        num_bins=10,
        **kwargs
    ):
        super(FastAPLoss, self).__init__(**kwargs)
        self.num_bins = int(num_bins)
        self.num_edges = self.num_bins + 1
        self.histogram_max = 4
        self.histogram_delta = self.histogram_max / self.num_bins
    
    def required_metric(self):
        return ["euclid_normalized"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple=None,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        dtype, device = metric_mat.dtype, metric_mat.device
        pos_mask = row_labels == col_labels
        neg_mask = ~ pos_mask
        if is_same_source:
            pos_mask.fill_diagonal_(False)
        pos_mask = pos_mask.type(dtype)
        neg_mask = neg_mask.type(dtype)

        N_pos = torch.sum(pos_mask, dim=1)
        safe_N = N_pos > 0
        if torch.sum(safe_N) == 0:
            return torch.sum(metric_mat * 0)
        
        # construct the histogram
        mid_points = (
            torch.linspace(0.0, self.histogram_max, steps=self.num_edges)
            .view(-1, 1, 1)
            .to(device)
            .type(dtype)
        )
        pulse = torch.relu(
            1 - torch.abs(metric_mat - mid_points) / self.histogram_delta
        )
        pos_hist = torch.sum(pulse * pos_mask, dim=2).t()
        neg_hist = torch.sum(pulse * neg_mask, dim=2).t()

        total_pos_hist = torch.cumsum(pos_hist, dim=1)
        total_hist = torch.cumsum(pos_hist + neg_hist, dim=1)

        h_pos_product = pos_hist * total_pos_hist
        safe_H = (h_pos_product > 0) & (total_hist > 0)
        if torch.sum(safe_H) > 0:
            FastAP = torch.zeros_like(pos_hist).to(device)
            FastAP[safe_H] = h_pos_product[safe_H] / total_hist[safe_H]
            FastAP = torch.sum(FastAP, dim=1)
            FastAP = FastAP[safe_N] / N_pos[safe_N]
            FastAP = 1 - FastAP
            return torch.mean(FastAP)
        else:
            return torch.sum(metric_mat * 0)