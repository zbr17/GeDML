import torch
import numpy as np 
import math

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class AngularLoss(BaseLoss):
    """
    paper: `Deep Metric Learning with Angular Loss <https://openaccess.thecvf.com/content_iccv_2017/html/Wang_Deep_Metric_Learning_ICCV_2017_paper.html>`_
    """
    def __init__(
        self,
        alpha=40,
        **kwargs
    ):
        super(AngularLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.sq_tan_alpha = math.tan(math.radians(self.alpha)) ** 2
    
    def required_metric(self):
        return ["cosine"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple,
        is_same_source=False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # get indices
        anchor_index, positive_index, keep_mask = self.get_pairs(row_labels, col_labels, indices_tuple)

        # compute loss
        ap_pair = metric_mat[anchor_index, positive_index].unsqueeze(1)
        ap_n_pair = (
            metric_mat[anchor_index,:] +
            metric_mat[positive_index,:]
        )
        f_apn = (
            (4 * self.sq_tan_alpha * ap_n_pair) -
            (2 * (1 + self.sq_tan_alpha) * ap_pair)
        )
        loss = l_f.logsumexp(f_apn, keep_mask=keep_mask, add_one=True, dim=1)
        loss = torch.mean(loss)

        return loss
    
    def get_pairs(self, row_labels, col_labels, indices_tuple):
        a1, p, a2, _ = l_f.split_indices(indices_tuple)
        if len(a1) == 0 or len(a2) == 0:
            return [None] * 4
        keep_mask = row_labels[a1, :] != col_labels
        return a1, p, keep_mask.byte()
