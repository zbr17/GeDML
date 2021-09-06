import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class LiftedStructureLoss(BaseLoss):
    """
    paper: `Deep Metric Learning via Lifted Structured Feature Embedding <https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Song_Deep_Metric_Learning_CVPR_2016_paper.html>`_
    """
    def __init__(
        self,
        neg_margin=1,
        pos_margin=0,
        **kwargs
    ):
        super(LiftedStructureLoss, self).__init__(**kwargs)
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin
    
    def required_metric(self):
        return ["euclid"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple=None,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        a1, p, a2, _ = l_f.split_indices(indices_tuple)
        pos_pair, neg_pair = l_f.indices_to_pairs(metric_mat, indices_tuple)
        dtype = metric_mat.dtype

        if len(a1) > 0 and len(a2) > 0:
            pos_pair = pos_pair.unsqueeze(1)
            n_per_p = (
                (a2.unsqueeze(0) == a1.unsqueeze(1))
                | (a2.unsqueeze(0) == p.unsqueeze(1))
            ).type(dtype)
            neg_pair = neg_pair * n_per_p
            keep_mask = ~ (n_per_p == 0)

            remaining_neg_margin = self.neg_margin - neg_pair
            remaining_pos_margin = pos_pair - self.pos_margin
            neg_pair_loss = l_f.logsumexp(
                remaining_neg_margin, keep_mask=keep_mask, add_one=False, dim=1
            )
            loss_per_pos_pair = neg_pair_loss + remaining_pos_margin
            loss_per_pos_pair = torch.relu(loss_per_pos_pair) ** 2
            loss_per_pos_pair /= 2
            loss = torch.mean(loss_per_pos_pair)
            return loss
        else:
            return torch.sum(metric_mat * 0)
