import torch
import torch.nn as nn
import math
from copy import deepcopy
from torchdistlog import logging
import torch.distributed as dist 

from ..base_collector import BaseCollector
from ...misc import utils
from ...models import BatchNormMLP

class SimSiamCollector(BaseCollector):
    """
    Paper: `Exploring Simple Siamese Representation Learning <https://arxiv.org/abs/2011.10566>`_

    This method use none of the following to learn meaningful representations:

    1. negative sample pairs;
    2. large batches;
    3. momentum encoders.

    And a stop-gradient operation plays an essential role in preventing collapsing.
    """
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(SimSiamCollector, self).__init__(*args, **kwargs) 
        self.predictor = BatchNormMLP(
            layer_size_list=[2048, 512, 2048],
            relu_list=[True, False],
            bn_list=[True, False]
        )
    
    def forward(self, data, embeddings, labels) -> tuple:
        """
        For simplicity, two data streams will be combined together and be passed through ``embeddings`` parameter. In function ``collect``, two data streams will be split (first half for first stream; second half for second stream).

        Args:
            data (torch.Tensor):
                A batch of key images (**not used**). size: :math:`B \\times C \\times H \\times W`
            embeddings (torch.Tensor):
                A batch of query embeddings. size: :math:`2B \\times dim`
            labels (torch.Tensor):
                Labels of the input. size: :math:`2B \\times 1`
        """
        # split two streams
        N = embeddings.size(0)
        assert N % 2 == 0
        z1, z2 = embeddings[:N//2], embeddings[N//2:]
        labels = labels[:N//2]

        # compute p
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        metric_mat = 0.5 * (
            self.metric(p1, z2.detach()) +
            self.metric(p2, z1.detach())
        )
        return (
            metric_mat,
            labels.unsqueeze(1),
            labels.unsqueeze(0),
            False
        )
