import torch
from ..base_loss import BaseLoss

class ProxyLoss(BaseLoss):
    """
    paper: `No Fuss Distance Metric Learning Using Proxies <https://openaccess.thecvf.com/content_iccv_2017/html/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.html>`_
    """
    def __init__(
        self,
        gamma,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
    
    def required_metric(self):
        return ["euclid"]
    
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
        exp = torch.nn.functional.softmax(
            self.gamma * metric_mat, dim=-1
        )
        exp = torch.sum(
            exp * pos_mask, dim=-1
        )
        exp = exp[torch.where(exp)[0]]
        loss = torch.mean( - torch.log(exp))
        return loss