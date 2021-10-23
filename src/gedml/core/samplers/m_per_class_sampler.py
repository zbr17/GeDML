import numpy as np 
from .base_sampler import BaseSampler

class MPerClassSampler(BaseSampler):
    """
    Give m samples per class. In the distributed mode, random seed should be asigned manually.

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
        *args,
        **kwargs
    ):
        super(MPerClassSampler, self).__init__(*args, **kwargs)
        self.labels = labels
        self.m = m 
        self.batch_size = batch_size

        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)
        
        if self.is_distributed:
            # if distributed, batch-size = num-processes * sub-batch-size
            self.batch_size = self.world_size * self.batch_size 

        self.init_indices()
    
    def init_indices(self):
        assert self.batch_size % self.m == 0
        
        # construct indices
        self.labels_set = np.unique(self.labels)
        self.labels_to_indices = {}
        for label in self.labels_set:
            self.labels_to_indices[label] = np.where(self.labels == label)[0]

        batch_class = self.batch_size // self.m
        self.num_iters = len(self.labels_set) // batch_class
        total_class = self.num_iters * batch_class
        self.total_size = self.m * total_class
        assert self.total_size >= self.batch_size

        if self.is_distributed:
            assert self.batch_size % (self.m * self.world_size) == 0
            self.num_samples = self.total_size // self.world_size
        else:
            self.num_samples = self.total_size

    def __len__(self):
        return self.num_samples
        
    def __genidx__(self):
        idx_list = [0] * self.total_size
        class_num = self.batch_size // self.m
        i = 0
        np.random.shuffle(self.labels_set)
        for idx in range(self.num_iters):
            curr_labels_set = self.labels_set[idx*class_num : (idx+1)*class_num]
            for label in curr_labels_set:
                curr_indices = self.labels_to_indices[label]
                idx_list[i:i+self.m] = np.random.choice(curr_indices, size=self.m, replace=False)
                i += self.m 
        return idx_list
