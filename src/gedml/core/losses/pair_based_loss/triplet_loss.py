import torch

from ...misc import loss_function as l_f 
from ..base_loss import BaseLoss

class TripletLoss(BaseLoss):
    """
    paper: `Learning local feature descriptors with triplets and shallow convolutional neural networks <https://www.researchgate.net/profile/Krystian_Mikolajczyk/publication/317192886_Learning_local_feature_descriptors_with_triplets_and_shallow_convolutional_neural_networks/links/5a038dad0f7e9beb1770c3c2/Learning-local-feature-descriptors-with-triplets-and-shallow-convolutional-neural-networks.pdf>`_
    """
    def __init__(
        self,
        margin=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
    
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
        pos_pair, neg_pair = l_f.indices_to_pairs(metric_mat, indices_tuple, assert_shape=[3])

        triplet_loss = torch.nn.functional.relu(pos_pair - neg_pair + self.margin)

        # mean_triplet_loss = torch.mean(triplet_loss)

        nonzero_triplet_loss = triplet_loss[torch.where(triplet_loss)[0]]
        loss = torch.mean(nonzero_triplet_loss)
        return loss

    
