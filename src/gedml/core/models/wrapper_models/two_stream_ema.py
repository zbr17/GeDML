import torch
import torch.nn as nn
from copy import deepcopy
from torchdistlog import logging
import torch.distributed as dist

from ...modules import WithRecorder
from ... import models
from ...misc import utils

class TwoStreamEMA(WithRecorder):
    def __init__(
        self,
        base_module: nn.Module,
        batch_shuffle: bool,
        m: float,
        *args,
        **kwargs
    ):
        super(TwoStreamEMA, self).__init__(*args, **kwargs)
        self.m = m
        self.query_model = base_module
        self.batch_shuffle = batch_shuffle
        self.key_model = deepcopy(base_module)

        self.initiate_models()
    
    def initiate_models(self):
        for param_k in self.key_model.parameters():
            param_k.requires_grad = False # not update by gradient

    @torch.no_grad()
    def _ema_update(self):
        for param_q, param_k in zip(self.query_model.parameters(), self.key_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x: torch.Tensor):
        """
        Batch shuffle, for making use of BatchNorm.
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
    
    def forward(self, data):
        """
        The first half of the data is the query branch (updated by gradient), 
        and the second half is the key branch (memory bank)
        """
        if self.training:
            return self._train_forward(data)
        else:
            return self._eval_forward(data)

    def _train_forward(self, data):
        """
        Training phase: use batch-shuffle.
        """
        # split data
        assert data.shape[0] % 2 == 0
        bs = data.shape[0] // 2
        data_q = data[:bs]
        data_k = data[bs:]

        # forward models
        out_q = self.query_model(data_q)
        with torch.no_grad(): # no gradient to keys
            self._ema_update() # update the key encoder

            # shuffle for making use of BN
            if dist.is_initialized() and self.batch_shuffle:
                data_k, idx_unshuffle = self._batch_shuffle_ddp(data_k)
            
            out_k = self.key_model(data_k)

            # undo shuffle
            if dist.is_initialized() and self.batch_shuffle:
                out_k = self._batch_unshuffle_ddp(out_k, idx_unshuffle)
        # concat out
        out = torch.cat([out_q, out_k], dim=0)
        return out
    
    def _eval_forward(self, data):
        """
        Evaluation phase
        """
        out = self.query_model(data)
        return out