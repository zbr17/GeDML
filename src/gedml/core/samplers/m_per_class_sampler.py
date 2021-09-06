import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np 

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
    ):
        # raise NotImplementedError()
        self.labels = labels
        self.m = m 
        self.batch_size = batch_size

        self.init_indices()
    
    def init_indices(self):
        # construct indices
        self.labels_to_indices = defaultdict(list)
        for i, label in enumerate(self.labels):
            self.labels_to_indices[label].append(i)
        for k, v in self.labels_to_indices.items():
            self.labels_to_indices[k] = np.array(v).astype('int')
        self.labels_set = list(self.labels_to_indices.keys())
        self.list_size = self.m * len(self.labels_set)

        # assert
        assert self.list_size >= self.batch_size
        assert self.batch_size % self.m == 0
        self.list_size -= self.list_size % self.batch_size
        self.num_iters = self.list_size // self.batch_size
    
    def __len__(self):
        return self.list_size
        
    def __iter__(self):
        idx_list = [0] * self.list_size
        i = 0
        for _ in range(self.num_iters):
            np.random.shuffle(self.labels_set)
            curr_labels_set = self.labels_set[:self.batch_size // self.m]
            for label in curr_labels_set:
                curr_indices = self.labels_to_indices[label]
                idx_list[i:i+self.m] = safe_random_choice(curr_indices, size=self.m)
                i += self.m 
        return iter(idx_list)
