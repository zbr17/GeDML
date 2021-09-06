import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class ContrastiveLoss(BaseLoss):
    """
    paper: `Learning a Similarity Metric Discriminatively, with Application to Face Verification <https://ieeexplore.ieee.org/abstract/document/1467314/>`_
    """
    def __init__(
        self,
        pos_margin=0,
        neg_margin=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

        to_record_list = [
            "mean_pos_pair_dist",
            "mean_neg_pair_dist",
            "mean_pos_loss",
            "mean_neg_loss",
            "nonzero_mean_pos_loss",
            "nonzero_mean_neg_loss"
        ]
        for item in to_record_list:
            self.add_recordable_attr(name=item)
    
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
        **kwargs,
    ) -> torch.Tensor:
        pos_pair, neg_pair = l_f.indices_to_pairs(metric_mat, indices_tuple)

        self.mean_pos_pair_dist = torch.mean(pos_pair)
        self.mean_neg_pair_dist = torch.mean(neg_pair)

        pos_loss = torch.nn.functional.relu(pos_pair - self.pos_margin)
        neg_loss = torch.nn.functional.relu(self.neg_margin - neg_pair)

        self.mean_pos_loss = torch.mean(pos_loss)
        self.mean_neg_loss = torch.mean(neg_loss)

        pos_loss = pos_loss[torch.where(pos_loss)[0]]
        neg_loss = neg_loss[torch.where(neg_loss)[0]]

        self.nonzero_mean_pos_loss = torch.mean(pos_loss)
        self.nonzero_mean_neg_loss = torch.mean(neg_loss)

        numerator = torch.sum(pos_loss) + torch.sum(neg_loss)
        denominator = len(pos_loss) + len(neg_loss)
        loss = numerator / denominator
        return loss
    
