from PIL import Image
import torch
import numpy as np 

def base_collate_fn(info_dict: list):
    assert isinstance(info_dict, list)
    elem = info_dict[0]
    assert isinstance(elem, dict)
    info_dict = {
        key: _concatenate_tensor(
            [item[key] for item in info_dict]
        )
        for key in elem
    }
    return info_dict

def _concatenate_tensor(batch: torch.Tensor):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # modified from: https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel) # TODO: What's Tensor.FloatStorage? 
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, np.int64):
        return torch.tensor(batch)
    else:
        raise TypeError("Wrong type {}".format(type(elem)))

class TwoCropsTransformWrapper(object):
    """
    Take two random crops of one image as the query and key.
    modified from: https://github.com/facebookresearch/moco
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, img):
        query = self.base_transform(img)
        key = self.base_transform(img)
        return {
            "data": query,
            "addition_data": key
        }

    @staticmethod
    # Contrastive representation learning collate fn
    def modify_info_dict(info_dict):
        info_dict.pop("id")
        # generate self-supervise labels
        data_len = len(info_dict["data"])
        info_dict["labels"] = torch.arange(data_len).repeat(1, 2).flatten()
        # concatenate two streams
        data_stream2 = info_dict.pop("addition_data")
        info_dict["data"] = torch.cat([info_dict["data"], data_stream2], dim=0)
        return info_dict
    
    @staticmethod
    def collate_fn(info_dict: dict):
        info_dict = base_collate_fn(info_dict)
        info_dict = TwoCropsTransformWrapper.modify_info_dict(info_dict)
        return info_dict

class DefaultTransformWrapper(object):
    """
    Default wrapper.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, img):
        img = self.base_transform(img)
        return {
            "data": img,
        }