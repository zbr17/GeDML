import torch

from .base_similarity_metric import BaseSimilarityMetric

class MoCoMetric(BaseSimilarityMetric):
    def __call__(self, queries, keys, queue):
        # positive matrix: Nx1
        pos_mat = torch.einsum("ij,ij->i", [queries, keys]).unsqueeze(-1)
        # negative matrix: NxK
        neg_mat = torch.einsum("ik,kj->ij", [queries, queue.t()])
        # concat
        matrix = torch.cat([pos_mat, neg_mat], dim=1)
        return matrix
         