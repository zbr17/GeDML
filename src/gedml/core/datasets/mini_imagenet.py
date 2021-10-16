import torch
import numpy as np 
from torchvision import datasets
import os 

from .base_dataset import BaseDataset

class MiniImageNet(BaseDataset):
    def _set_dataset_info(self):
        if self.assert_num_classes is None:
            if self.phase == "train":
                self.assert_num_classes = 80
            else:
                self.assert_num_classes = 20
        if self.assert_num_samples is None:
            if self.phase == "train":
                self.assert_num_samples = 48000
            else:
                self.assert_num_samples = 12000

    def init_dataset(self):
        self._set_dataset_info()
        self.root = os.path.join(self.root, "mini_imagenet")
        img_folder = os.path.join(self.root, self.phase)
        self.dataset = datasets.ImageFolder(img_folder)
        self.labels = np.array([b for (a,b) in self.dataset.imgs])
        assert len(np.unique(self.labels)) == self.assert_num_classes
        assert self.__len__() == self.assert_num_samples
    
    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.dataset)
    
    def get_image_label(self, idx):
        return self.dataset[idx]
