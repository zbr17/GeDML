import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class CircleLoss(BaseLoss):
    """
    modified from: https://github.com/KevinMusgrave/pytorch-metric-learning

    paper: `Circle Loss: A Unified Perspective of Pair Similarity Optimization <http://openaccess.thecvf.com/content_CVPR_2020/html/Sun_Circle_Loss_A_Unified_Perspective_of_Pair_Similarity_Optimization_CVPR_2020_paper.html>`_
    """
    def __init__(
        self,
        m=0.4,
        gamma=80,
        **kwargs
    ):
        super(CircleLoss, self).__init__(**kwargs)
        self.m = m
        self.gamma = gamma
        self.op = 1 + m
        self.on = - m
        self.delta_p = 1 - m
        self.delta_n = m
    
    def required_metric(self):
        return ["cosine"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        pos_mask = (row_labels == col_labels)
        neg_mask = ~ pos_mask
        if is_same_source:
            pos_mask.fill_diagonal_(False)
        
        tmp_mat = torch.zeros_like(metric_mat)
        pos_pair = metric_mat[pos_mask]
        neg_pair = metric_mat[neg_mask]

        # construct matrix
        tmp_mat[pos_mask] = (
            - self.gamma
            * torch.relu(self.op - pos_pair.detach())
            * (pos_pair - self.delta_p)
        )
        tmp_mat[neg_mask] = (
            self.gamma
            * torch.relu(neg_pair.detach() - self.on)
            * (neg_pair - self.delta_n)
        )

        # compute logsumexp
        se_pos = l_f.sumexp(
            tmp_mat, keep_mask=pos_mask, dim=1
        )
        se_neg = l_f.sumexp(
            tmp_mat, keep_mask=neg_mask, dim=1
        )

        loss = torch.log(1 + se_pos * se_neg)
        zero_rows = torch.where(
            (torch.sum(pos_mask, dim=1) != 0) & 
            (torch.sum(neg_mask, dim=1) != 0)
        )[0]
        loss = loss[zero_rows]
        if len(loss) == 0:
            loss = torch.sum(metric_mat * 0)
        else:
            loss = torch.mean(loss)
        return loss

