import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class MarginLoss(BaseLoss):
    """
    paper: `Sampling Matters in Deep Embedding Learning <http://openaccess.thecvf.com/content_iccv_2017/html/Wu_Sampling_Matters_in_ICCV_2017_paper.html>`_
    """
    def __init__(
        self,
        alpha=0.2,
        beta=1.2,
        nu=0,
        num_classes=100,
        beta_constant=False,
        is_similarity=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.nu = nu
        self.num_classes = num_classes
        self.is_similarity = is_similarity
        self.beta_constant = beta_constant
        self.beta = (
            beta if beta_constant
            else torch.nn.Parameter(torch.ones(num_classes) * beta)
        )
    
    def required_metric(self):
        return ["euclid"]
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple,
        is_same_source=False,
        *args,
        **kwargs
    ) -> torch.Tensor:
        a1, p, a2, n = l_f.split_indices(indices_tuple)
        pos_pair, neg_pair = l_f.indices_to_pairs(metric_mat, indices_tuple)

        if self.beta_constant:
            beta_pos = beta_neg = self.beta
        else:
            beta_pos = self.beta[row_labels[a1].flatten()]
            beta_neg = self.beta[row_labels[a2].flatten()]

        if not self.is_similarity:
            pos_loss = torch.nn.functional.relu(self.alpha + pos_pair - beta_pos)
            neg_loss = torch.nn.functional.relu(self.alpha - neg_pair + beta_neg)
        else:
            pos_loss = torch.nn.functional.relu(beta_pos + self.alpha - pos_pair)
            neg_loss = torch.nn.functional.relu(neg_pair - beta_neg + self.alpha)

        # mean_triplet_loss = torch.mean(triplet_loss)

        pos_loss = pos_loss[torch.where(pos_loss)[0]]
        neg_loss = neg_loss[torch.where(neg_loss)[0]]

        numerator = torch.sum(pos_loss) + torch.sum(neg_loss)
        denominator = len(pos_loss) + len(neg_loss)
        loss = numerator / denominator
        return loss