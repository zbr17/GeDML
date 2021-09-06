import logging
import shutil
from sys import path
import pandas as pd 
import numpy as np 
import os

from gedml.launcher.misc import utils
from . import utils as utils_recorder

from .base_info_getter import BaseInfoGetter
from .base_info_writer import BaseInfoWriter

class BaseRecorder:
    """
    Recorder can search and save two forms of information:

    1. Attributions of module. If the module is inherited from ``WithRecorder`` or the module maintain a **to-record** list named 'to_record_list'.
    2. Dictionary with specific format:

    .. code:: python
    
        {
            table_name:
                item_name:
                    value
                    ...
        }

    Args:
        root (str):
            Path where to save the statistics and checkpoint.
        hint_if_exist (bool):
            Whether to hint if the folder is existed (in case that user delete the folder by mistake)
        delete_old_folder (bool):
            Whether to delete the old folder if the folder is existed.
    """
    def __init__(
        self,
        project_name="test",
        root="./",
        group_name="main",
        hint_if_exist=True,
        delete_old_folder=True,
        is_resume=False,
        use_wandb=True,
        initiate_create=True,
    ):
        self.project_name = project_name
        self.root = root
        self.group_name = group_name
        self.exp_name = os.path.basename(self.root)
        self.hint_if_exist = hint_if_exist
        self.delete_old_folder = delete_old_folder
        self.is_resume = is_resume
        self.use_wandb = use_wandb

        if initiate_create:
            self.create_exp_root()
            self.create_group_folders(group_name)

    def create_exp_root(self):
        self.root = utils_recorder.create_folder(
            path=self.root,
            is_resume=self.is_resume,
            hint_if_exist=self.hint_if_exist,
            delete_old_folders=self.delete_old_folder
        )

    def create_group_folders(self, group_name):
        sub_root_name = os.path.join(self.root, group_name)
        self.getter = BaseInfoGetter()
        self.writer = BaseInfoWriter(
            project_name=self.project_name,
            root=sub_root_name,
            exp_name=self.exp_name + "_{}".format(group_name),
            hint_if_exist=self.hint_if_exist,
            delete_old_folder=self.delete_old_folder,
            is_resume=self.is_resume,
            use_wandb=self.use_wandb
        )
    
    def delete_folders(self, group_name):
        sub_root_name = os.path.join(self.root, group_name)
        if os.path.exists(sub_root_name):
            shutil.rmtree(sub_root_name)
            logging.info("Delete group folder: {}".format(sub_root_name))

    
    def log_config(self, config_dict):
        self.writer.log_config(config_dict)
    
    def get_data(self, *args, **kwargs):
        """
        Extract the information from a module.

        Args:
            data (dict or obj):
                Dictionary of obj with **TO_RECORD_LIST** attribute.
            name (str):
                Specific name.
        
        Returns:
            tuple: (dict, int)

            1. output_dict (dict): information dictionary to be saved.
            2. step (int): step counted by ``recorder``.
        """
        return self.getter.get_data(*args, **kwargs)
    
    def update(self, *args, **kwargs):
        """
        Update record.

        Args:
            data (dict):
                Dictionary to be recorded.
            step (int):
                Step (optional).
        """
        self.writer.update(*args, **kwargs)
    
    def save_models(self, *args, **kwargs):
        """
        Save models.

        Args:
            obj (module):
                Modules with **TO_SAVE_LIST**.
            step (int):
                Step (or epoch)
            best (bool):
                Whether to be the best checkpoint.
            delete_old (bool):
                Whether to delete the old checkpoint.
        """
        self.writer.save_models(*args, **kwargs)
    
    def load_models(self, *args, **kwargs):
        """
        Load models.

        Args:
            obj (module):
                Modules with **TO_SAVE_LIST**.
            step (int):
                Step (or epoch).
            device (device):
                The computing device to load.
        """
        self.writer.load_models(*args, **kwargs)
    
    def search_best_epoch(self, table_name, primary_key, maximum=True):
        """
        Search best epoch according given evaluation indicator.

        Args:
            table_name (str):
                The csv file.
            primary_key (str): 
                The primary_key to be searched.
            maximum (bool):
                Whether to regard the maximum to be the best.
        
        Returns:
            tuple: (best step, best metric)
        """
        table_path = os.path.join(self.writer.csv_path, table_name)
        df = pd.read_csv(table_path)
        # get primary metric
        selected_key = df[primary_key]
        if maximum:
            best_idx = selected_key.idxmax()
        else:
            best_idx = selected_key.idxmin()
        return df['step'][best_idx], df[primary_key][best_idx]
    