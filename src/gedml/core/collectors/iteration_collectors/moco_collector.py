import torch
import math
from copy import deepcopy
from torchdistlog import logging
import torch.distributed as dist 
import torch.nn.functional as F
from ..base_collector import BaseCollector
from ...misc import utils

class MoCoCollector(BaseCollector):
    """
    Paper: `Momentum Contrast for Unsupervised Visual Representation Learning <https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html>`_

    Use Momentum Contrast (MoCo) for unsupervised visual representation learning. This code is modified from: https://github.com/facebookresearch/moco. In this paper, a dynamic dictionary with a queue and a moving-averaged encoder are built.

    Args:
        embeddings_dim (int): 
            dimension of embeddings. default: 128
        bank_size (int): 
            size of the memory bank. default: 65536
        m (float): 
            weight of moving-average. default: 0.999
        T (float): 
            coefficient of softmax
    """
    def __init__(
        self,
        embeddings_dim=128,
        bank_size=65536,
        T=0.07,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs) 
        self.embeddings_dim = embeddings_dim
        self.bank_size = bank_size
        self.T = T
        self.initiate_params()
    
    def initiate_params(self):
        """
        Initiate memory bank.
        """
        self.register_buffer("queue", torch.randn(self.bank_size, self.embeddings_dim))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # gather keys before updating queue
        if dist.is_initialized():
            keys, = utils.distributed_gather_objects(keys)
        batch_size = keys.shape[0]

        # get pointer
        ptr = int(self.queue_ptr)
        assert self.bank_size % batch_size == 0 # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:(ptr+batch_size), :] = keys
        ptr = (ptr + batch_size) % self.bank_size # move pointer
        self.queue_ptr[0] = ptr
    
    def forward(self, embeddings) -> tuple:
        """
        Maintain a large memory bank to boost learning performance.

        Args:
            data (torch.Tensor):
                A batch of key images. size: :math:`B \\times C \\times H \\times W`
            embeddings (torch.Tensor):
                A batch of query embeddings. size: :math:`B \\times dim`
        """
        # compute key features
        device = embeddings.device
        batch_size_all = embeddings.shape[0]
        if dist.is_initialized():
            world_size = dist.get_world_size()
            batch_size = batch_size_all // world_size // 2
            sub_embeddings = embeddings.view(dist.get_world_size(), 2 * batch_size, -1)[dist.get_rank()]
        else:
            batch_size = batch_size_all // 2
            sub_embeddings = embeddings
        embedding_q = sub_embeddings[:batch_size]
        embedding_k = sub_embeddings[batch_size:]
        
        # compute matrix
        metric_mat = self.metric(
            embedding_q,
            embedding_k.clone().detach(),
            self.queue.clone().detach()
        )
        # apply temperature
        metric_mat /= self.T 

        # generate labels 
        row_labels = torch.arange(batch_size).to(device).unsqueeze(-1)
        neg_labels = torch.arange(batch_size, self.bank_size + batch_size).unsqueeze(0).repeat(batch_size, 1).to(device)
        col_labels = torch.cat(
            [
                row_labels,
                neg_labels
            ],
            dim=-1
        )

        # dequeue and enqueue
        self._dequeue_and_enqueue(embedding_k)

        is_same_source = False
        return (
            metric_mat,
            row_labels,
            col_labels,
            is_same_source
        )