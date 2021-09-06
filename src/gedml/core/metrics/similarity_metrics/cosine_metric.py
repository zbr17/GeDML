import torch

from .base_similarity_metric import BaseSimilarityMetric

class CosineMetric(BaseSimilarityMetric):
    def __call__(self, row_input, col_input=None):
        if col_input is None:
            col_input = row_input
        return torch.matmul(row_input, col_input.t())
         