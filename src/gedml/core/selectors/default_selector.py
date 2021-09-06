from .base_selector import BaseSelector

class DefaultSelector(BaseSelector):
    """
    Do nothing selector.
    """
    def forward(
        self, 
        metric_mat, 
        row_labels, 
        col_labels, 
        is_same_source=False
    ) -> tuple:
        """
        Do nothing.
        """
        indices_tuple, weight = None, None
        return metric_mat, row_labels, col_labels, is_same_source, indices_tuple, weight