import os
import sys
import shutil
from torchdistlog import logging
import torch

def get_first_key_of_dict(data: dict) -> str:
    first_key = list(data.keys())[0]
    return first_key

def create_folder(
    path, 
    is_resume=False, 
    hint_if_exist=True, 
    delete_old_folders=True
):
    """
    Create folder.

    Args:
        path (str):
            Path to be created.
        is_resume (bool):
            Whether to set resume mode. (default: False)
        hint_if_exist (bool):
            Hind if the folder is existed (It only works when delete_old_folders is True)
        delete_old_folders (bool):
            Whether to delete old folder. If it's True, old folder with the same name will be deleted.
            Otherwise, a "(x)" like suffix will be used to mark the new folder.
    
    Returns:
        str: Return the path if the path was modified.
    """
    if os.path.exists(path):
        if is_resume:
            logging.info("RESUME mode: not create new folder - {}!".format(path))
        else:
            if delete_old_folders:
                if hint_if_exist:
                    delete_choice = input("{path} exists! DELETE or NOT (Y/N): ".format(path=path))
                    if delete_choice != 'Y':
                        sys.exit(-1)
                shutil.rmtree(path)
                logging.info("{path} is RE-created!".format(path=path))
            else:
                ### use suffix to distinguish these folders with the same name.
                # check the current number of folders with the same name.
                suffix = "_({})"
                index = 2
                newpath = path + suffix.format(index)
                while(os.path.exists(newpath)):
                    index += 1
                    newpath = path + suffix.format(index)
                os.makedirs(newpath)
                path = newpath
                logging.info("New {path} is created!".format(path=path))
    else:
        os.makedirs(path)
        logging.info("New {path} is created!".format(path=path))
    
    return path

def meta_call_factory(obj, ordered_list, prefix, description_str=None, **kwargs):
    for func_suffix in ordered_list:
        func_name = prefix + func_suffix
        func_obj = getattr(obj, func_name)
        func_obj(**kwargs)
        if isinstance(description_str, str):
            logging.info(description_str.format(func_obj.__name__))

def get_value(data: (torch.Tensor or ...)):
    if isinstance(data, torch.Tensor):
        return data.item()
    elif isinstance(data, (int, float)):
        return data
    else:
        raise TypeError('Invalid data type: {}'.format(type(data)))
