import torch
import numpy as np 
from pandas import read_csv
import os
from PIL import Image
from .base_dataset import BaseDataset

class OnlineProducts(BaseDataset):
    """
    `Stanford Online Products <https://cvgl.stanford.edu/projects/lifted_struct/>`_
    """
    def _set_dataset_info(self):
        if self.assert_num_classes is None:
            if self.phase == "train":
                self.assert_num_classes = 11318
            else:
                self.assert_num_classes = 11316
        if self.assert_num_samples is None:
            if self.phase == "train":
                self.assert_num_samples = 59551
            else:
                self.assert_num_samples = 60502

    def init_dataset(self):
        self._set_dataset_info()
        self.root = os.path.join(self.root, "online_products")
        info_folder = os.path.join(self.root, 'Info_Files')
        img_folder = os.path.join(self.root, 'images')
        self.img_paths, self.labels = [], []
        self.init_info_file_name()

        curr_file = read_csv(
            os.path.join(info_folder, self.info_file_name),
            delim_whitespace=True, 
            header=0
        ).values
        self.img_paths.extend(
            [os.path.join(img_folder, name) 
            for name in list(curr_file[:, 3])]
        )
        self.labels.extend(list(curr_file[:, 1] - 1))
        
        self.labels = np.array(self.labels)
        assert len(np.unique(self.labels)) == self.assert_num_classes
        assert self.__len__() == self.assert_num_samples
    
    def init_info_file_name(self):
        if self.phase == "train":
            self.info_file_name = "Ebay_train.txt"
        elif self.phase == "test":
            self.info_file_name = "Ebay_test.txt"
        else:
            raise KeyError(
                "Invalid dataset phase: {} / {}".format(
                    self.phase,
                    self.__class__.__name__
                )
            )

    def get_labels(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)

    def get_image_label(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        return img, label