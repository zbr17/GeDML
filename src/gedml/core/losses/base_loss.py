import torch
import torch.nn.functional as F 
from abc import abstractmethod, ABCMeta

from ..modules import WithRecorder

class BaseLoss(WithRecorder, metaclass=ABCMeta):
    """
    Base loss module. The output of this module will be wrapped with "FINISH" flag which indicates the output doesn't need to be further processed.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple=None,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        loss = self.compute_loss(
            metric_mat,
            row_labels,
            col_labels,
            indices_tuple=indices_tuple,
            weights=weights,
            is_same_source=is_same_source
        )
        return loss

    @abstractmethod
    def compute_loss(
        self,
        metric_mat,
        row_labels,
        col_labels,
        indices_tuple=None,
        weights=None,
        is_same_source=False,
    ) -> torch.Tensor:
        """
        Compute loss value.

        Args:
            metric_mat (torch.Tensor):
                Metric matrix.
            row_labels (torch.Tensor):
                Labels of matrix rows.
            col_labels (torch.Tensor):
                Labels of matrix columns.
            indices_tuple (dict):
                Dict that has two keys: "tuples" and "flags"
            weights (torch.Tensor):
                Can be element-wised, tuple-wised etc.
            is_same_source (bool):


        Returns:
            torch.Tensor: Final loss value (a tensor value).
        """
        return 0
    
    @abstractmethod
    def required_metric(self) -> list:
        return []