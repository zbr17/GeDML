import torch
import numpy as np 
from torchvision import datasets
import os
import scipy.io as sio
import PIL.Image

from .base_dataset import BaseDataset

class Cars196(BaseDataset):
    """
    `Cars196 <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_

    Use "cars_annos.mat" to load the images.
    """
    def _set_dataset_info(self):
        if self.assert_num_classes is None:
            self.assert_num_classes = 98
        if self.assert_num_samples is None:
            if self.phase == "train":
                self.assert_num_samples = 8054
            else:
                self.assert_num_samples = 8131

    def init_dataset(self):
        self._set_dataset_info()

        self.root = os.path.join(self.root, "cars196")
        self.annos_file = os.path.join(self.root, "cars_annos.mat")
        annos_file = sio.loadmat(self.annos_file)

        if self.phase == "train":
            classes = range(0, 98)
        elif self.phase == "test":
            classes = range(98, 196)
        else:
            raise KeyError("Invalid self.phase!")
        
        paths = [
            item[0][0] for item 
            in annos_file["annotations"][0]
        ]
        labels = [
            int(item[5][0] - 1) for item 
            in annos_file["annotations"][0]
        ]

        self.paths, self.labels = [], []
        for path, label in zip(paths, labels):
            if label in classes:
                self.paths.append(os.path.join(self.root, path))
                self.labels.append(label)
        self.labels = np.array(self.labels)

        assert len(np.unique(self.labels)) == self.assert_num_classes
        assert self.__len__() == self.assert_num_samples

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.paths)
    
    def get_image_label(self, idx):
        # load image
        img = PIL.Image.open(self.paths[idx])
        # convert gray to rgb
        if len(list(img.split())) == 1:
            img = img.convert("RGB")
        # load label
        label = self.labels[idx]
        return img, label

# class Cars196(BaseDataset):
#     """
#     `Cars196 <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_

#     NOTE: This old implementation has been depreciated!
#     """
#     def init_dataset(self):
#         self.root = os.path.join(self.root, "cars196")
#         img_folder = os.path.join(self.root, self.phase)
#         self.dataset = datasets.ImageFolder(img_folder)
#         self.labels = np.array([b for (a, b) in self.dataset.imgs])
#         assert len(np.unique(self.labels)) == self.assert_num_classes
#         assert self.__len__() == self.assert_num_samples

#     def get_labels(self):
#         return self.labels

#     def __len__(self):
#         return len(self.dataset)
    
#     def get_image_label(self, idx):
#         return self.dataset[idx]