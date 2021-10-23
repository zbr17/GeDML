from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import numpy as np 

from abc import abstractmethod, ABCMeta

class BaseSampler(Sampler, metaclass=ABCMeta):
    """
    Base class of sampler. In the distributed mode, random seed should be asigned manually.
    """
    def __init__(
        self,
        seed=0,
        rank=None,
        world_size=None,
        is_distributed=False,
    ):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.epoch = 0

        if self.is_distributed:
            if self.rank is None:
                self.rank = dist.get_rank()
            if self.world_size is None:
                self.world_size = dist.get_world_size()
        
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __genidx__(self) -> list:
        pass

    def __iter__(self):
        if self.is_distributed:
            np.random.seed(self.seed + self.epoch)
        
        idx_list = self.__genidx__()
        if self.is_distributed:
            idx_list = np.array(idx_list).reshape(-1, self.m)
            idx_list = idx_list[self.rank: len(idx_list) : self.world_size].flatten().tolist()
        return iter(idx_list)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
    def choice(self, input_list, size):
        replace = len(input_list) < size
        return np.random.choice(input_list, size=size, replace=replace)
