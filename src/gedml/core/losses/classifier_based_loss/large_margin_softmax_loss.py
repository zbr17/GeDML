import torch
import scipy.special
import math

from ..base_loss import BaseLoss

class LargeMarginSoftmaxLoss(BaseLoss):
    """
    modified from: https://github.com/KevinMusgrave/pytorch-metric-learning

    paper: `Large-Margin Softmax Loss for Convolutional Neural Networks <https://www.jmlr.org/proceedings/papers/v48/liud16.pdf>`_
    """
    def __init__(
        self,
        margin=4,
        scale=1,
        **kwargs
    ):
        super(LargeMarginSoftmaxLoss, self).__init__(**kwargs)
        self.margin = margin
        self.scale = scale
        self.initiate_margin()

    def required_metric(self):
        return ["cosine"]
    
    def initiate_margin(self):
        self.margin = int(self.margin)
        self.max_n = self.margin // 2
        ## For the trigonometric multiple-angle formula ##
        self.n_range = torch.Tensor([
            n for n in range(0, self.max_n + 1)
        ])
        self.margin_choose_n = torch.Tensor([
            scipy.special.binom(self.margin, 2 * n) for n in self.n_range
        ])
        self.cos_powers = torch.Tensor([
            self.margin - (2 * n) for n in self.n_range
        ])
        self.alternating = torch.Tensor([
            (-1) ** n for n in self.n_range
        ])
    
    def get_cos_with_margin(self, cosine):
        cosine = cosine.unsqueeze(1)
        for attr in ["n_range", "margin_choose_n", "cos_powers", "alternating"]:
            setattr(self, attr, getattr(self, attr).to(cosine.device))
        cos_powered = cosine ** self.cos_powers
        sin_powered = (1 - cosine ** 2) ** self.n_range
        terms = (
            self.alternating * self.margin_choose_n * cos_powered * sin_powered
        )
        return torch.sum(terms, dim=1)
    
    def get_target_mask(self, metric_mat, labels):
        batch_size = labels.size(0)
        mask = torch.zeros_like(metric_mat, device=metric_mat.device)
        mask[torch.arange(batch_size), labels] = 1
        return mask
    
    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1, 1))
        return angles
    
    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        cos_with_margin = self.get_cos_with_margin(cosine_of_target_classes)
        angles = self.get_angles(cosine_of_target_classes)
        with torch.no_grad():
            k = (
                angles / (math.pi / self.margin)
            ).floor()
        return ((-1) ** k) * cos_with_margin - (2 * k)
    
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple=None,
        is_same_source=False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        dtype, device = metric_mat.dtype, metric_mat.device
        mask = self.get_target_mask(metric_mat, row_labels.squeeze())
        cosine_of_target_classes = metric_mat[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes 
        )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
        logits = metric_mat + (mask * diff)
        # TODO: lack scaling function
        loss = torch.nn.functional.cross_entropy(logits * self.scale, row_labels.squeeze())

        return loss
    

    
