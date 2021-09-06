import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class MultiSimilarityLoss(BaseLoss):
    """
    paper: `Multi-Similarity Loss With General Pair Weighting for Deep Metric Learning <https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.html>`_
    """
    def __init__(
        self,
        alpha=2,
        beta=50,
        base=0.5,
        **kwargs
    ):
        super(MultiSimilarityLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.base = base
        
    def required_metric(self):
        return ["cosine"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple=None,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        # get masks
        pos_mask = row_labels == col_labels
        neg_mask = ~ pos_mask
        if is_same_source:
            pos_mask.fill_diagonal_(False)

        # compute loss
        pos_loss = (1.0 / self.alpha) * l_f.logsumexp(
            - self.alpha * (metric_mat - self.base), keep_mask=pos_mask, add_one=True, dim=1
        )
        neg_loss = (1.0 / self.beta) * l_f.logsumexp(
            self.beta * (metric_mat - self.base), keep_mask=neg_mask, add_one=True, dim=1
        )
        loss = torch.mean(pos_loss) + torch.mean(neg_loss)

        return loss