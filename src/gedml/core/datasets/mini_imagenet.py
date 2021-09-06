import torch
import numpy as np 
from torchvision import datasets
import os 

from .base_dataset import BaseDataset

class MiniImageNet(BaseDataset):
    def init_dataset(self):
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
