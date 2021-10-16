from numpy.lib.arraysetops import isin
import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np 
import torch.distributed as dist

def safe_random_choice(input_list, size):
    replace = len(input_list) < size
    return np.random.choice(input_list, size=size, replace=replace)

class MPerClassSampler(Sampler):
    """
    Give m samples per class.

    Args:
        labels (np.ndarray):
            Ground truth of datasets
        m (int):
            M samples per class
        batch_size (int): 
            Batch size must be an interger multiple of m
    """
    def __init__(
        self,
        labels,
        m,
        batch_size,
        seed=0,
        rank=None,
        world_size=None,
        is_distributed=False,
    ):
        # raise NotImplementedError()
        self.labels = labels
        self.m = m 
        self.batch_size = batch_size
        self.seed = 0
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.epoch = 0

        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        if self.is_distributed:
            if self.rank is None:
                self.rank = dist.get_rank()
            if self.world_size is None:
                self.world_size = dist.get_world_size()
            # if distributed, batch-size = num-processes * sub-batch-size
            self.batch_size = self.world_size * self.batch_size 

        self.init_indices()
    
    def init_indices(self):
        # construct indices
        self.labels_set = np.unique(self.labels)
        self.labels_to_indices = {}
        for label in self.labels_set:
            self.labels_to_indices[label] = np.where(self.labels == label)[0]

        self.total_size = self.m * len(self.labels_set)

        # assert
        assert self.total_size >= self.batch_size
        assert self.batch_size % self.m == 0
        if self.is_distributed:
            assert self.batch_size % (self.m * self.world_size) == 0
        self.total_size -= self.total_size % self.batch_size
        self.num_iters = self.total_size // self.batch_size

        if self.is_distributed:
            self.num_samples = self.total_size // self.world_size
        else:
            self.num_samples = self.total_size

    
    def __len__(self):
        return self.num_samples
        
    def __iter__(self):
        if self.is_distributed:
            np.random.seed(self.seed + self.epoch)
        
        idx_list = [0] * self.total_size
        class_num = self.batch_size // self.m
        i = 0
        np.random.shuffle(self.labels_set)
        for idx in range(self.num_iters):
            curr_labels_set = self.labels_set[idx*class_num : (idx+1)*class_num]
            for label in curr_labels_set:
                curr_indices = self.labels_to_indices[label]
                idx_list[i:i+self.m] = safe_random_choice(curr_indices, size=self.m)
                i += self.m 
        if self.is_distributed:
            idx_list = np.array(idx_list).reshape(-1, self.m)
            idx_list = idx_list[self.rank:len(idx_list):self.world_size].flatten()
        return iter(idx_list)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
