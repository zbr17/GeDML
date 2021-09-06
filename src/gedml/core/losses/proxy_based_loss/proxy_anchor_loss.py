import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class ProxyAnchorLoss(BaseLoss):
    """
    paper: `Proxy Anchor Loss for Deep Metric Learning <http://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html>`_
    """
    def __init__(
        self,
        margin=0.1,
        alpha=32,
        **kwargs
    ):
        super(ProxyAnchorLoss, self).__init__(**kwargs)
        self.alpha = alpha
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
        pos_mask = row_labels == col_labels
        neg_mask = ~ pos_mask

        with_pos_proxies = torch.where(torch.sum(pos_mask, dim=0) != 0)[0]

        pos_term = l_f.logsumexp(
            - self.alpha * (metric_mat - self.margin),
            keep_mask=pos_mask,
            add_one=True,
            dim=0
        ).squeeze()
        neg_term = l_f.logsumexp(
            self.alpha * (metric_mat + self.margin),
            keep_mask=neg_mask,
            add_one=True,
            dim=0
        ).squeeze()
        
        if len(with_pos_proxies) == 0:
            pos_loss = torch.sum(metric_mat * 0)
        else:
            pos_loss = torch.mean(pos_term[with_pos_proxies])
        neg_loss = torch.mean(neg_term)

        return pos_loss + neg_loss