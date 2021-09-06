import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class PosPairLoss(BaseLoss):
    """
    Designed for SimSiam.

    paper: `Exploring Simple Siamese Representation Learning <https://arxiv.org/abs/2011.10566>`_
    """
    def __init__(
        self,
        **kwargs
    ):
        super(PosPairLoss, self).__init__(**kwargs)
    
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
        pos_pair = metric_mat[pos_mask]
        loss = torch.mean( - pos_pair)
        return loss