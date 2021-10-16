from torchdistlog import logging
import numpy as np 

import torch.distributed as dist 
import torch

from torchdistlog import logging

try: 
    import faiss
except ModuleNotFoundError:
    logging.warning("Faiss Package Not Found!")

from ...launcher.misc.utils import distributed_gather_objects

"""
(Faiss) to multi gpu devices
"""

def index_cpu_to_gpu_multiple(index, resources=None, co=None, gpu_ids=None):
    assert isinstance(gpu_ids, list)
    if resources is None:
        resources = [faiss.StandardGpuResources() for i in gpu_ids]
    v_res = faiss.GpuResourcesVector()
    v_dev = faiss.IntVector()
    if gpu_ids is None:
        gpu_ids = list(range(len(resources)))
    for i, res in zip(gpu_ids, resources):
        v_dev.push_back(i)
        v_res.push_back(res)
    index = faiss.index_cpu_to_gpu_multiple(v_res, v_dev, index, co)
    index.referenced_objects = resources
    return index

"""
Some random number generator
"""
def multi_center_generator(sample_num_per_class, class_num, dim=2, scale=0.1):
    loc = np.random.rand(class_num, dim)
    scale = scale * np.ones((class_num, dim))
    data = []
    for i in range(class_num):
        data.append(np.random.normal(loc=loc[i, :], scale=scale[i, :], size=(sample_num_per_class, dim)))
    data = np.vstack(data).astype('float32')
    label = np.arange(class_num).reshape(-1, 1)
    label = np.tile(label, (1, sample_num_per_class)).reshape(-1).astype('float32')
    return data, label
