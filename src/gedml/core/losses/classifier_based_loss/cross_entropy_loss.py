import torch
import torch.nn.functional as F 
from ..base_loss import BaseLoss

class CrossEntropyLoss(BaseLoss):
    """
    paper: `Momentum Contrast for Unsupervised Visual Representation Learning <http://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html>`_

    Cross-entropy loss designed for MoCo.
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
    
    def required_metric(self):
        return ["moco"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        is_same_source=False,
        *args,
        **kwargs
    ) -> torch.Tensor:
        dtype, device = metric_mat.dtype, metric_mat.device
        pos_mask = (row_labels == col_labels).type(dtype).to(device)
        pos_index = torch.where(pos_mask)[1]

        loss = F.cross_entropy(metric_mat, pos_index)
        return loss