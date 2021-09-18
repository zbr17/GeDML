import os
import logging
import csv
import yaml
import wandb
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch
from . import utils
from ..config.setting.recorder_setting import CSV_FOLDER, MODEL_FOLDER, BOARD_FOLDER, WANDB_FOLDER, TO_SAVE_LIST, DEVICE, STEP_MODEL_SUFFIX, BEST_MODEL_SUFFIX, LOAD_EXCEPT_FUNC

class BaseInfoWriter:
    """
    This writer will save information into csv files and tensorboard. And manage model saving and loading.
        - root
        - csv
        - model
        - board
    """
    def __init__(
        self,
        project_name,
        root,
        exp_name,
        delete_old_folder=True,
        hint_if_exist=True,
        is_resume=False,
        use_wandb=True,
        params_to_save=None,
        link_config=None,
        pipeline_to_save=None,
    ): 
        self.project_name = project_name
        self.root = root
        self.exp_name = exp_name
        self.delete_old_folder = delete_old_folder
        self.hint_if_exist = hint_if_exist
        self.is_resume = is_resume
        self.use_wandb = use_wandb
        self.params_to_save = params_to_save
        self.link_config = link_config
        self.pipeline_to_save = pipeline_to_save

        self.csv_path, self.model_path, self.board_path, self.wandb_path = None, None, None, None

        self.folders_list = ['csv', 'model', 'board'] # can be modified
        self.update_ordered_list = ['csv', 'board'] # can be modified
        self.update_function_prefix = '_update_'
        self.init_ordered_list = ['board', 'model']
        self.init_function_prefix = '_init_'

        if self.use_wandb:
            self.folders_list += ['wandb']
            self.update_ordered_list += ['wandb']
            self.init_ordered_list += ['wandb']

        self.init_folders()
        self.init_update_handler()
    
    """
    INITIATE
    """

    def _meta_path_factory(self, item):
        path = os.path.join(self.root, globals()["{}_FOLDER".format(item.upper())])
        setattr(self, "{}_path".format(item), path)
        utils.create_folder(path, is_resume=self.is_resume, delete_old_folders=self.delete_old_folder)
        return None
    
    def init_folders(self):
        # create root folder
        self.root = utils.create_folder(self.root, is_resume=self.is_resume, hint_if_exist=self.hint_if_exist, delete_old_folders=self.delete_old_folder)
        # save config file
        self.save_params()
        # save pipeline flow chart
        self.save_pipeline()
        # create sub folders
        for item in self.folders_list:
            self._meta_path_factory(item)
    
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
                logging.warn("GraphViz isn't installed! Pipeline flow chart generation FAILED!")

    
    def init_update_handler(self):
        utils.meta_call_factory(
            obj=self,
            ordered_list=self.init_ordered_list,
            prefix=self.init_function_prefix,
            description_str="Initiate update handler by {}"
        )

    def _init_board(self):
        self.board_handler = SummaryWriter(log_dir=self.board_path)
    
    def _init_model(self):
        self.model_handler = BaseModelHandler(self.model_path)
        self.save_models = self.model_handler.save_models
        self.load_models = self.model_handler.load_models
    
    def _init_wandb(self):
        wandb.init(
            project=self.project_name,
            name=self.exp_name,
            dir=self.wandb_path,
            settings=wandb.Settings(start_method="fork")
        )
    
    def log_config(self, config_dict):
        if self.use_wandb:
            wandb.config.update(config_dict)
            logging.info("Wandb config updated!")
        else:
            logging.info("NOT use wandb!")
    
    """
    UPDATE 
    """

    def update(self, data: dict, step: int=None):
        """
        This is the primary API for BaseInfoWriter class.

        Args:
            data        (dict):     Dict
            step        (int):      Step (optional)
        """
        utils.meta_call_factory(
            obj=self,
            ordered_list=self.update_ordered_list,
            prefix=self.update_function_prefix,
            data=data,
            step=step
        )
    
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
    
    def __del__(self):
        wandb.finish()

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
    
    def load_models(self, obj, step=None, device=None, model_path=None):
        """
        TODO: resume from the checkpoint with max number if "best" models doesn't exist.
        """
        model_path = (
            self.model_path
            if model_path is None
            else model_path
        )
        max_index_list = [-2]
        to_save_list = getattr(obj, TO_SAVE_LIST, {})
        device = getattr(obj, DEVICE, torch.device('cpu')) if device is None else device
        for to_save_name in to_save_list:
            to_save_item = getattr(obj, to_save_name, {})
            for k, v in to_save_item.items():
                import glob
                item_search = glob.glob(os.path.join(model_path, to_save_name + "_" + k+"*"))
                try:
                    index_list = [item.split(".")[0].split("_")[-1] for item in item_search]
                    index_list = [int(item) if item!="best" else -2 for item in index_list]
                    max_index = max(index_list)
                except:
                    logging.warn("Can't recognize the last training epoch!")
                    max_index = -2
                max_index_list.append(max_index)
                if len(item_search) < 1:
                    logging.warn("{} does not exist!".format(model_path))
                curr_model_name = (
                    to_save_name + "_" + k + STEP_MODEL_SUFFIX.format(int(step)) 
                    if step is not None else 
                    to_save_name + "_" + k + BEST_MODEL_SUFFIX
                )
                curr_model_path = os.path.join(model_path, curr_model_name)
                if curr_model_path not in item_search:
                    curr_model_path = item_search[0]
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



    