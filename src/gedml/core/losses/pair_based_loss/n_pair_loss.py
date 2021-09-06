import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class NPairLoss(BaseLoss):
    """
    Work with NPairSelector (Recommend)

    paper: `Improved deep metric learning with multi-class n-pair loss objective <http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf>`_
    """
    def __init__(
        self,
        **kwargs
    ):
        super(NPairLoss, self).__init__(**kwargs)
    
    def required_metric(self):
        return ["dot_product"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple=None,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        anchor_idx, positive_idx = l_f.get_unique_indices(indices_tuple, row_labels.squeeze())

        if len(anchor_idx) == 0:
            return torch.sum(metric_mat * 0)
        
        targets = torch.arange(len(anchor_idx)).to(metric_mat.device)
        sub_mat = metric_mat[anchor_idx, :][:, positive_idx]
        loss = torch.nn.functional.cross_entropy(sub_mat, targets)
        return loss