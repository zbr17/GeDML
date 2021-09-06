import torch
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import numpy as np 
from copy import deepcopy

def safe_random_choice(input_list, size):
    replace = len(input_list) < size
    return np.random.choice(input_list, size=size, replace=replace)

class MPerClassFullSampler(Sampler):
    """
    Give m samples per class (Try to cover all the samples).

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
        self.min_label_num = min([len(v) for v in self.labels_to_indices.values()])
        self.list_size = self.min_label_num * len(self.labels_set)

        # assert
        assert self.list_size >= self.batch_size
        assert self.batch_size % self.m == 0
        self.list_size -= self.list_size % self.batch_size
        self.num_iters = self.list_size // self.batch_size
    
    def __len__(self):
        return self.list_size
        
    def __iter__(self):
        num_group = int(self.min_label_num / self.m)
        num_label = int(self.batch_size / self.m)

        idx_list = [0] * self.list_size
        store_dict = {
            label: safe_random_choice(self.labels_to_indices[label], self.min_label_num)
            for label in self.labels_set
        }
        store_dict = {
            k: [
                v[idx*self.m:(idx+1)*self.m]
                for idx in range(num_group)
            ]
            for k, v in store_dict.items()
        }
        self.labels_set = np.array(self.labels_set)
        total_label_list = []
        for _ in range(num_group):
            label_set = deepcopy(self.labels_set.tolist())
            np.random.shuffle(label_set)
            total_label_list += label_set

        list_idx = 0
        for i in range(self.num_iters):
            # avaliable curr-labels-set
            curr_labels_set = total_label_list[i*num_label : (i+1)*num_label]

            # get batch-size samples
            for label in curr_labels_set:
                curr_indices = store_dict[label].pop(0)
                idx_list[list_idx:list_idx+self.m] = curr_indices
                list_idx += self.m 
        return iter(idx_list)

if __name__ == "__main__":
    labels_num = np.random.choice([60, 61, 62, 63], size=(75,), replace=True)
    labels = []
    for i in range(75):
        labels += [i] * labels_num[i]
    sampler = MPerClassFullSampler(labels=labels, m=4, batch_size=32)
    iter_ = iter(sampler)
    pass