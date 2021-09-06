import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class SoftTripleLoss(BaseLoss):
    """
    paper: `SoftTriple Loss: Deep Metric Learning Without Triplet Sampling <https://openaccess.thecvf.com/content_ICCV_2019/html/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.html>`_
    """
    def __init__(
        self,
        centers_per_class=10,
        num_classes=100,
        la=20,
        gamma=0.1,
        margin=0.01,
        **kwargs
    ):
        super(SoftTripleLoss, self).__init__(**kwargs)
        self.centers_per_class = centers_per_class
        self.num_classes = num_classes
        self.la = la 
        self.gamma = gamma
        self.margin = margin
    
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
        metric_to_centers = metric_mat.view(
            -1, self.num_classes, self.centers_per_class
        )
        prob = torch.nn.functional.softmax(
            metric_to_centers * self.gamma, dim=2
        )
        metric_to_classes = torch.sum(
            prob * metric_to_centers, dim=2
        )
        margin = torch.zeros_like(metric_to_classes)
        margin[torch.arange(margin.shape[0]), row_labels.squeeze()] = self.margin
        loss = torch.nn.functional.cross_entropy(
            self.la * (metric_to_classes - margin), row_labels.squeeze()
        )
        return loss