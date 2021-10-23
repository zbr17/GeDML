from collections import defaultdict
from os import replace
import numpy as np 
from copy import deepcopy
from .base_sampler import BaseSampler

class MPerClassFullSampler(BaseSampler):
    """
    Give m samples per class (Try to cover all the samples). In the distributed mode, random seed should be asigned manually.

    Args:
        labels (np.ndarray):
            Ground truth of datasets
        m (int):
            M samples per class
        batch_size (int): 
            Batch size must be an interger multiple of m
        samples_per_class (int or str):
            Indicate the numbers of samples from classes. (int, "max" or "min")
    """
    def __init__(
        self,
        labels,
        m,
        batch_size,
        samples_per_class="max",
        *args,
        **kwargs
    ):
        super(MPerClassFullSampler, self).__init__(*args, **kwargs)
        self.labels = labels
        self.m = m
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)
        
        if self.is_distributed:
            # if distributed, batch-size = num-processes * sub-batch-size
            self.batch_size = self.world_size * self.batch_size
        
        self.init_indices()
    
    def _gen_samples_per_class(self):
        if isinstance(self.samples_per_class, (int, float)):
            self.samples_per_class = int(self.samples_per_class)
        elif self.samples_per_class == "max":
            self.samples_per_class = max([len(v) for v in self.labels_to_indices.values()])
        elif self.samples_per_class == "min":
            self.samples_per_class = min([len(v) for v in self.labels_to_indices.values()])
        else:
            raise KeyError("Invalid value of 'samples_per_class'")
        
        assert self.batch_size % self.m == 0
        self.tuples_per_class = self.samples_per_class // self.m
        self.samples_per_class = self.tuples_per_class * self.m
        self.tuples_per_batch = self.batch_size // self.m
        self.num_batches = (self.tuples_per_class * len(self.labels_set)) // self.tuples_per_batch
        self.total_size = self.num_batches * self.batch_size
        assert self.total_size >= self.batch_size
        if self.is_distributed:
            assert self.batch_size % (self.m * self.world_size) == 0
            self.num_samples = self.total_size // self.world_size
        else:
            self.num_samples = self.total_size

    def init_indices(self):
        assert self.batch_size % self.m == 0

        # construct indices
        self.labels_set = np.unique(self.labels)
        self.labels_to_indices = {}
        self.labels_to_lengths = {}
        for label in self.labels_set:
            self.labels_to_indices[label] = np.where(self.labels == label)[0]
            self.labels_to_lengths[label] = len(self.labels_to_indices[label])
        self._gen_samples_per_class()

################################
    
    def __len__(self):
        return self.num_samples
        
    def __genidx__(self):
        store_dict = {
            label: np.random.choice(
                self.labels_to_indices[label], 
                self.samples_per_class,
                self.samples_per_class > self.labels_to_lengths[label])
            for label in self.labels_set
        }
        store_dict = {
            k: [
                v[idx*self.m:(idx+1)*self.m]
                for idx in range(self.tuples_per_class)
            ]
            for k, v in store_dict.items()
        }

        total_label_list = np.repeat(
            self.labels_set.reshape(1, -1),
            self.tuples_per_class,
            axis=0
        ).flatten()
        for idx in range(self.num_batches):
            start = idx * self.tuples_per_batch
            end = (idx + 1) * self.tuples_per_batch
            np.random.shuffle(
                total_label_list[start: end]
            )
        total_label_list = total_label_list[:(self.tuples_per_batch * self.num_batches)].reshape(-1, self.tuples_per_batch)
        
        list_idx = 0
        idx_list = [0] * self.total_size
        for curr_label_set in total_label_list:
            # get batch-size samples
            for label in curr_label_set:
                curr_indices = store_dict[label].pop(0)
                idx_list[list_idx : list_idx+self.m] = curr_indices
                list_idx += self.m
        
        return idx_list

if __name__ == "__main__":
    labels_num = np.random.choice([60, 61, 62, 63], size=(75,), replace=True)
    labels = []
    for i in range(75):
        labels += [i] * labels_num[i]
    sampler = MPerClassFullSampler(labels=labels, m=4, batch_size=32)
    iter_ = iter(sampler)
    pass