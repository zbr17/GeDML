import torch

from .base_distance_metric import BaseDistanceMetric

class SNRMetric(BaseDistanceMetric):
    def __call__(self, row_input, col_input=None):
        if col_input is None:
            col_input = row_input
        denominator = torch.var(row_input, dim=1)
        numerator = torch.var(row_input.unsqueeze(1) - col_input, dim=2)
        return numerator / (denominator.unsqueeze(1))