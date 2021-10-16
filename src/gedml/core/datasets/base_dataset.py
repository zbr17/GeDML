from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    Base class of dataset. 

    Args:
        root (str):
            Root path of datasets
        phase (str):
            "train" or "test"
        assert_num_classes (int):
            Check whether the data is completely loaded.
        assert_num_samples (int):
            Check whether the data is completely loaded.
        transform (transform module):
            Image transform (augmentation). Default: None.
    """
    def __init__(self, root, phase, assert_num_classes=None, assert_num_samples=None, transform=None):
        self.root = root
        self.phase = phase
        self.assert_num_classes = assert_num_classes
        self.assert_num_samples = assert_num_samples
        self.transform = transform
        # initiate
        self.init_dataset()

    def set_transform(self, transform):
        self.transform = transform
    
    @abstractmethod
    def get_labels(self):
        pass

    @abstractmethod
    def init_dataset(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx):
        img, label = self.get_image_label(idx)
        if self.transform is not None:
            img_dict = self.transform(img)
        else:
            raise NotImplementedError("No Transform!")
        info_dict = {
            "labels": label,
            "id": idx
        }
        info_dict.update(img_dict)
        return info_dict
    
    @abstractmethod
    def get_image_label(self, idx):
        pass
    