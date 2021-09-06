import torch
from abc import ABCMeta, abstractmethod

from ..modules import WithRecorder

class BaseSelector(WithRecorder, metaclass=ABCMeta):
    """
    Base class of ``selectors``.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(
        self, 
        metric_mat, 
        row_labels, 
        col_labels, 
        is_same_source=False
    ) -> tuple:
        """
        Args:
            metric_mat (torch.Tensor):
                Metric matrix.
            row_labels (torch.Tensor):
                Labels of rows.
            col_labels (torch.Tensor): 
                Labels of columns.
            is_same_source (bool):
                Whether the two data streams are from the same source.
        
        Returns:
            tuple: Five type of elements:

            1. metric_mat (torch.Tensor): Metric matrix.
            2. labels_row (torch.Tensor): Labels of rows.
            3. labels_col (torch.Tensor): Labels of columns.
            4. is_same_source (bool): Whether the two tensors are from the same source.
            5. indices_tuple (dict): Dict that has two key: "tuples" and "flags"
            6. weights (torch.Tensor): Weights.
        """
        indices_tuple, weights = None, None
        return (
            metric_mat, 
            row_labels, 
            col_labels, 
            is_same_source, 
            indices_tuple, 
            weights
        )
