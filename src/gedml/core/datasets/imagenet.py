import torch
import tarfile
from torchdistlog.tqdm import tqdm
import numpy as np 
from torchdistlog import logging
from pandas import read_csv
import os
import re
import shutil
from torchvision import datasets
from .base_dataset import BaseDataset

class ImageNet(BaseDataset):
    """
    `ImageNet <http://www.image-net.org/>`_
    """
    def _set_dataset_info(self):
        if self.assert_num_classes is None:
            if self.phase == "train":
                self.assert_num_classes = 1000
            else:
                self.assert_num_classes = 1000
        if self.assert_num_samples is None:
            if self.phase == "train":
                self.assert_num_samples = 1281167
            else:
                self.assert_num_samples = 50000

    def init_dataset(self):
        self._set_dataset_info()
        self.root = os.path.join(self.root, "imagenet")
        if self.phase == "train":
            self.init_train_dataset()
        elif self.phase == "test":
            self.init_test_dataset()
        else:
            raise KeyError(
                "Invalid dataset phase: {}".format(
                    self.phase
                )
            )
    
    """
    train set
    """

    def init_train_dataset(self):
        self.img_src = os.path.join(self.root, "imagenet_train")
        self.img_folder = os.path.join(self.root, "train")
        if not os.path.exists(self.img_folder):
            logging.info("PID: {} - Start to extract train iamges: {}".format(
                os.getpid(),
                self.__class__.__name__
            ))
            self.extract_train_img(self.img_src, self.img_folder)
        self.dataset = datasets.ImageFolder(self.img_folder)
        self.labels = np.array([b for (a,b) in self.dataset.imgs])
        assert len(np.unique(self.labels)) == self.assert_num_classes
        assert self.__len__() == self.assert_num_samples
    
    def extract_train_img(self, img_src, img_dst):
        tar_list = os.listdir(img_src)
        for file_name in tqdm(tar_list):
            tar = tarfile.open(os.path.join(img_src, file_name), "r")
            tar.extractall(path=os.path.join(img_dst, file_name.split('.')[0]))
            tar.close()
        
    """
    val set (test)
    """

    def init_test_dataset(self):
        self.img_src = os.path.join(self.root, "imagenet_val")
        self.img_folder = os.path.join(self.root, "test")
        self.info_file_path = os.path.join(
            self.root, 
            "ILSVRC2012_devkit_t12", 
            "data", 
            "ILSVRC2012_validation_ground_truth.txt"
        )
        if not os.path.exists(self.img_folder):
            logging.info("PID: {} - Start to extract test images: {}".format(
                os.getpid(),
                self.__class__.__name__
            ))
            self.extract_test_img(self.img_src, self.img_folder)
        self.dataset = datasets.ImageFolder(self.img_folder)
        self.labels = np.array([b for (a,b) in self.dataset.imgs])
        assert len(np.unique(self.labels)) == self.assert_num_classes
        assert self.__len__() == self.assert_num_samples
    
    def extract_test_img(self, img_src, img_dst):
        img_name_list = os.listdir(img_src)
        img_name_dict = {
            int(re.findall("\d+", name)[1]): name
            for name in img_name_list
        }
        info_file = read_csv(
            self.info_file_path,
            delim_whitespace=True,
            header=None
        ).values
        img_label_dict = {
            int(idx+1): int(item)
            for idx, item in enumerate(info_file)
        }
        for k, curr_img_label in tqdm(img_label_dict.items()):
            curr_img_name = img_name_dict[k]
            img_label_folder = os.path.join(img_dst, str(curr_img_label))
            if not os.path.exists(img_label_folder):
                os.makedirs(img_label_folder)
            img_src_path = os.path.join(img_src, curr_img_name)
            img_dst_path = os.path.join(img_label_folder, curr_img_name)
            shutil.copyfile(img_src_path, img_dst_path)
            
    
    def get_labels(self):
        return self.labels 
    
    def __len__(self):
        return len(self.dataset)
    
    def get_image_label(self, idx):
        return self.dataset[idx]