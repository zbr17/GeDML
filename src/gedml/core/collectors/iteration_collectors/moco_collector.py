import torch
import math
from copy import deepcopy
import logging
import torch.distributed as dist 
from ..base_collector import BaseCollector
from ...misc import utils

class MoCoCollector(BaseCollector):
    """
    Paper: `Momentum Contrast for Unsupervised Visual Representation Learning <https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html>`_

    Use Momentum Contrast (MoCo) for unsupervised visual representation learning. This code is modified from: https://github.com/facebookresearch/moco. In this paper, a dynamic dictionary with a queue and a moving-averaged encoder are built.

    Args:
        query_trunk (torch.nn.Module): default: ResNet50
        query_embedder (torch.nn.Module): multi-layer perceptron
        embeddings_dim (int): dimension of embeddings. default: 128
        bank_size (int): size of the memory bank. default: 65536
        m (float): weight of moving-average. default: 0.999
        T (float): coefficient of softmax
    """
    def __init__(
        self,
        query_trunk,
        query_embedder,
        embeddings_dim=128,
        bank_size=65536,
        m=0.999,
        T=0.07,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs) 
        self.embeddings_dim = embeddings_dim
        self.bank_size = bank_size
        self.m = m
        self.T = T
        self.key_trunk = deepcopy(query_trunk)
        self.key_embedder = deepcopy(query_embedder)
        self.initiate_params()
    
    def initiate_params(self):
        """
        Cancel the gradient of key_trunk and key_embedder
        """
        for param_k in self.key_trunk.parameters():
            param_k.requires_grad = False # not update by gradient
        for param_ke in self.key_embedder.parameters():
            param_ke.requires_grad = False # not update by gradient

        self.register_buffer("queue", torch.randn(self.bank_size, self.embeddings_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self, models_dict: dict):
        """
        Momentum update of the key model (encoder)
        """
        for param_q, param_k in zip(models_dict["trunk"].parameters(), self.key_trunk.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_qe, param_ke in zip(models_dict["embedder"].parameters(), self.key_embedder.parameters()):
            param_ke.data = param_ke.data * self.m + param_qe.data * (1. - self.m)
    
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
        ptr = (ptr + batch_size) % self.bank_size
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: torch.Tensor):
        """
        Batch shuffle, for making use of BatchNorm.

        Note:
            Only support DistributedDataParallel (DDP) model.
        """
        # gather from all gpus
        device = x.device
        batch_size_this = x.shape[0]
        x_gather, = utils.distributed_gather_objects(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(device)

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x: torch.Tensor, idx_unshuffle: torch.Tensor):
        """
        Undo batch shuffle.

        Note:
            Only support DistributedDataParallel (DDP) model.
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather, = utils.distributed_gather_objects(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    @torch.no_grad()
    def update(self, trainer):
        """
        Update key encoder using moving-average.
        """
        with torch.no_grad():
            self._momentum_update_key_encoder(
                trainer.models
            )
    
    def forward(self, data, embeddings) -> tuple:
        """
        Maintain a large memory bank to boost unsupervised learning performance.

        Args:
            data (torch.Tensor):
                A batch of key images. size: :math:`B \\times C \\times H \\times W`
            embeddings (torch.Tensor):
                A batch of query embeddings. size: :math:`B \\times dim`
        """
        # compute key features
        device = embeddings.device
        batch_size = embeddings.shape[0]
        if dist.is_initialized():
            world_size = dist.get_world_size()
            batch_size = batch_size // world_size
            sub_embeddings = embeddings.view(dist.get_world_size(), batch_size, -1)[dist.get_rank()]
        else:
            sub_embeddings = embeddings
        
        with torch.no_grad(): # no gradient to keys
            # shuffle for making use of BN
            if dist.is_initialized():
                data, idx_unshuffle = self._batch_shuffle_ddp(data)
            
            # get key embeddings
            keys = self.get_key_embeddings(data) # keys: N x C

            # undo shuffle
            if dist.is_initialized():
                keys = self._batch_unshuffle_ddp(keys, idx_unshuffle)
        
        # compute matrix
        metric_mat = self.metric(
            sub_embeddings,
            keys.clone().detach(),
            self.queue.clone().detach()
        )

        # apply temperature
        metric_mat /= self.T 

        # generate labels 
        # TODO: 为什么不用管queue中的正样本？
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
        self._dequeue_and_enqueue(keys)

        is_same_source = False
        return (
            metric_mat,
            row_labels,
            col_labels,
            is_same_source
        )
    
    def get_key_embeddings(self, data):
        trunk_output = self.key_trunk(data)
        embeddings = self.key_embedder(trunk_output)
        return embeddings