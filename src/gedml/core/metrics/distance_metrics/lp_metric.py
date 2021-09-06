import torch

from .base_distance_metric import BaseDistanceMetric

class LpMetric(BaseDistanceMetric):
    def __init__(self, p=2):
        self.p = p

    def __call__(self, row_input, col_input=None):
        if col_input is None:
            col_input = row_input
        return torch.cdist(row_input, col_input, p=self.p)
