import shutil
from sys import prefix
import pandas as pd 
import numpy as np 
import os
from copy import deepcopy
import csv
import yaml
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torchdistlog import logging

from . import utils
from ..config.setting.recorder_setting import TO_RECORD_LIST, TO_SAVE_LIST, DEVICE, STEP_MODEL_SUFFIX, BEST_MODEL_SUFFIX, LOAD_EXCEPT_FUNC

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
        params_to_save=None,
        link_config=None,
        pipeline_to_save=None,
        initiate_create=True,
        is_distributed=False,
        output_rank=0,
    ):
        self.project_name = project_name
        self.meta_root = root
        self.group_name = group_name
        self.exp_name = os.path.basename(self.meta_root)
        self.hint_if_exist = hint_if_exist
        self.delete_old_folder = delete_old_folder
        self.is_resume = is_resume
        self.use_wandb = use_wandb
        self.params_to_save = params_to_save
        self.link_config = link_config
        self.pipeline_to_save = pipeline_to_save
        self.is_distributed = is_distributed

        self.output_rank = output_rank
        self.dict_counts = {}
        self.csv_path = None
        self.model_path = None
        self.board_path = None
        self.wandb_path = None
        self.root = None
        self.model_handler = None

        
        self.update_ordered_list = ['csv', 'board'] # can be modified
        self.update_function_prefix = '_update_'

        if self.use_wandb:
            self.update_ordered_list += ['wandb']

        self.if_record = False
        # check if distributed
        if not self.is_distributed:
            self.if_record = True
        else:
            self.hint_if_exist = False # is hard to "input a Y" when distributed
            self.curr_rank = dist.get_rank()
            if self.curr_rank == self.output_rank:
                self.if_record = True
        
        if initiate_create:
            self.create_meta_root_folder()
            self.create_group_folders(group_name)
    
    def __del__(self):
        wandb.finish()

    ### Function group 1: Create folders and initiate handlers

    def save_params(self):
        if isinstance(self.params_to_save, dict):
            param_path = os.path.join(self.root, "param")
            utils.create_folder(param_path, hint_if_exist=False, delete_old_folders=True)
            for class_name, item_list in self.link_config.items():
                sub_param_path = os.path.join(param_path, class_name)
                os.makedirs(sub_param_path)
                for item_info in item_list:
                    # get yaml file name
                    item_name = utils.get_first_key_of_dict(item_info)
                    yaml_name = item_info[item_name]
                    sub_save_path = os.path.join(sub_param_path, item_name + "-" + yaml_name)
                    # get dictionary to save
                    sub_dict = deepcopy(self.params_to_save[class_name][item_name])
                    object_name = sub_dict.pop("type")
                    sub_dict = {object_name: sub_dict}
                    # save
                    with open(sub_save_path, mode="w", encoding="utf-8") as f:
                        yaml.dump(sub_dict, f, allow_unicode=True)
    
    def save_pipeline(self):
        if self.pipeline_to_save is not None:
            try:
                from graphviz import Digraph
                file_path = os.path.join(self.root, "pipeline")
                gz = Digraph(
                    name="pipeline",
                    node_attr={"style": "rounded", "shape": "box"}
                )
                for start, end, note in self.pipeline_to_save:
                    gz.edge(start, end, note)
                gz.render(file_path)
                logging.info("Pipeline flow chart is stored in {}".format(file_path))
            except:
                logging.warning("GraphViz isn't installed! Pipeline flow chart generation FAILED!")

    def create_meta_root_folder(self):
        if self.if_record:
            self.meta_root = utils.create_folder(
                path=self.meta_root,
                is_resume=self.is_resume,
                hint_if_exist=self.hint_if_exist,
                delete_old_folders=self.delete_old_folder
            )

    def create_group_folders(self, group_name):
        self.root = os.path.join(self.meta_root, group_name)
        self.exp_name = "{}_{}".format(self.exp_name, group_name)
        self.csv_path = os.path.join(self.root, "csv")
        self.board_path = os.path.join(self.root, "board")
        self.model_path = os.path.join(self.root, "model")
        self.wandb_path = os.path.join(self.root, "wandb")
        self._init_model()

        if self.if_record:
            def create_folder(path):
                path = utils.create_folder(
                    path=path,
                    is_resume=self.is_resume,
                    hint_if_exist=self.hint_if_exist,
                    delete_old_folders=self.delete_old_folder
                ) 
                return path
            # create folders and initiate update handler
            create_folder(self.root)
            create_folder(self.csv_path)
            create_folder(self.model_path)
            create_folder(self.board_path)
            self._init_board()
            if self.use_wandb:
                create_folder(self.wandb_path)
                self._init_wandb()

            # save config file
            self.save_params()
            # save pipeline flow chart
            self.save_pipeline()
    
    def _init_board(self):
        self.board_handler = SummaryWriter(log_dir=self.board_path)
    
    def _init_model(self):
        self.model_handler = BaseModelHandler(self.model_path)
    
    def _init_wandb(self):
        wandb.init(
            project=self.project_name,
            name=self.exp_name,
            dir=self.wandb_path,
            settings=wandb.Settings(start_method="fork")
        )
    
    def delete_folders(self, group_name):
        sub_root_name = os.path.join(self.meta_root, group_name)
        if os.path.exists(sub_root_name):
            shutil.rmtree(sub_root_name)
            logging.info("Delete group folder: {}".format(sub_root_name))

    ### Function group 2: Parse and write records

    # Parse objects

    def assert_obj(self, obj):
        to_record_list = getattr(obj, TO_RECORD_LIST, None)
        assert to_record_list is not None, \
            'obj must have attribute: {}!'.format(TO_RECORD_LIST)
        return to_record_list
    
    def get_description(self, name, spliter="-"):
        description = spliter.join([name, 'attribute'])
        return description
    
    def convert_obj_to_dict(self, obj, name=None):
        """

        """
        to_record_list = self.assert_obj(obj)
        name = (
            obj.__class__.__name__
            if name is None else name
        )
        output = {}
        for record_item in to_record_list:
            output[record_item] = utils.get_value(
                getattr(obj, record_item, 0)
            )
        return {
            self.get_description(name): output
        }
    
    def increase_counts(self, key):
        if key not in self.dict_counts:
            self.dict_counts[key] = 0 
        else:
            self.dict_counts[key] += 1
    
    def get_data(self, data: (dict or ...), name=None):
        """
        Extract information from the input module.

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
        if self.if_record:
            # extract data to dict
            output = (
                self.convert_obj_to_dict(data, name)
                if not isinstance(data, dict) else data
            )
            assert len(list(output.keys())) < 2, \
                'first-level keys must be less than 2 in output dict'
            # increase counts
            primary_key = utils.get_first_key_of_dict(output)
            self.increase_counts(primary_key)
            return output, self.dict_counts[primary_key]
        else:
            return None, None
    
    # Write records

    def log_config(self, config_dict):
        if self.if_record:
            if self.use_wandb:
                wandb.config.update(config_dict)
                logging.info("Wandb config updated!")
            else:
                logging.info("NOT use wandb!")
    
    def _update_csv(self, data, step=None):
        table_name = utils.get_first_key_of_dict(data)
        table_name_csv = table_name + '.csv'
        table_path = os.path.join(self.csv_path, table_name_csv)
        data[table_name]['step'] = step
        if not os.path.exists(table_path):
            with open(table_path, mode='w', newline='') as f:
                f_csv = csv.DictWriter(f, fieldnames=list(data[table_name].keys()))
                f_csv.writeheader()
                f_csv.writerow(data[table_name])
        else:
            with open(table_path, mode='a', newline='') as f:
                f_csv = csv.DictWriter(f, fieldnames=list(data[table_name].keys()))
                f_csv.writerow(data[table_name])
        data[table_name].pop('step')

    def _update_board(self, data, step=None):
        primary_key = utils.get_first_key_of_dict(data)
        for k, v in data[primary_key].items():
            full_key = primary_key + '/' + k
            self.board_handler.add_scalar(full_key, utils.get_value(v), global_step=step)
    
    def _update_wandb(self, data, step=None):
        primary_key = utils.get_first_key_of_dict(data)
        wandb.log(data={primary_key+'/step': step})
        for k, v in data[primary_key].items():
            full_key = primary_key + '/' + k
            wandb.log(data={full_key: utils.get_value(v)})
    
    def update(self, data: dict, step: int=None):
        """
        Update record.

        Args:
            data (dict):
                Dictionary to be recorded.
            step (int):
                Step (optional).
        """
        if self.if_record:
            utils.meta_call_factory(
                obj=self,
                ordered_list=self.update_ordered_list,
                prefix=self.update_function_prefix,
                data=data,
                step=step
            )
    
    ### Function group 3: Save and load models

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
        if self.if_record:
            self.model_handler.save_models(*args, **kwargs)
    
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
        return self.model_handler.load_models(*args, **kwargs)
    
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
        table_path = os.path.join(self.csv_path, table_name)
        df = pd.read_csv(table_path)
        # get primary metric
        selected_key = df[primary_key]
        if maximum:
            best_idx = selected_key.idxmax()
        else:
            best_idx = selected_key.idxmin()
        return df['step'][best_idx], df[primary_key][best_idx]

class BaseModelHandler:
    def __init__(self, model_path):
        self.model_path = model_path

    def delete_models(self):
        models_list = os.listdir(self.model_path)
        for model_name in models_list:
            if 'best' not in model_name:
                model_item_path = os.path.join(self.model_path, model_name)
                os.remove(model_item_path)
    
    def save_models(self, obj, step, best=False, delete_old=True):
        """
        Args:
            best (bool)
        """
        if delete_old: self.delete_models()
        to_save_list = getattr(obj, TO_SAVE_LIST, {})
        for to_save_name in to_save_list:
            to_save_item = getattr(obj, to_save_name, {})
            assert isinstance(to_save_item, dict), "{} must a dictionary!".format(to_save_name)
            for k, v in to_save_item.items():
                curr_model_name = to_save_name + "_" + k + STEP_MODEL_SUFFIX.format(int(step))
                curr_model_path = os.path.join(self.model_path, curr_model_name)
                torch.save(v.state_dict(), curr_model_path)
                logging.info('{} is saved in {}'.format(curr_model_name, curr_model_path))
                if best:
                    best_model_name = to_save_name + "_" + k + BEST_MODEL_SUFFIX
                    best_model_path = os.path.join(self.model_path, best_model_name)
                    torch.save(v.state_dict(), best_model_path)
                    logging.info('{} is saved in {}'.format(best_model_name, best_model_path))
    
    def load_models(self, obj, step="best", device=None, model_path=None):
        """
        TODO: resume from the checkpoint with max number if "best" models doesn't exist.

        Args:
            step: str or int (default: "best")
            int, "max" or "best"
        """
        model_path = (
            self.model_path
            if model_path is None
            else model_path
        )
        max_index_list = [-2]
        to_save_list = getattr(obj, TO_SAVE_LIST, {})
        device = (
            getattr(obj, DEVICE, torch.device('cpu'))
            if device is None else device
        )
        for to_save_name in to_save_list:
            to_save_item = getattr(obj, to_save_name, {})
            for k, v in to_save_item.items():
                import glob
                item_search = glob.glob(os.path.join(model_path, to_save_name + "_" + k+"*"))
                assert len(item_search) > 0, \
                    "{} does not exist!".format(model_path)
                
                index_list = [item.split(".")[0].split("_")[-1] for item in item_search]
                index_list = [int(item) if item!="best" else -2 for item in index_list]
                max_index = max(index_list)
                max_index_list.append(max_index)
                
                if step == "best":
                    suffix = BEST_MODEL_SUFFIX
                elif step == "max":
                    suffix = STEP_MODEL_SUFFIX.format(int(max_index))
                else:
                    suffix = STEP_MODEL_SUFFIX.format(int(step))
                curr_model_name = "{}_{}".format(to_save_name, k+suffix)
                curr_model_path = os.path.join(model_path, curr_model_name)

                try:
                    v.load_state_dict(torch.load(curr_model_path, map_location=device))
                except:
                    state_dict = torch.load(curr_model_path, map_location=device)
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for p_k, p_v in state_dict.items():
                        name = LOAD_EXCEPT_FUNC(p_k)
                        new_state_dict[name] = p_v
                    v.load_state_dict(new_state_dict)
                    v = v.to(device)
                logging.info('{} is loaded from {}'.format(curr_model_name, curr_model_path))
        return max(max_index_list)