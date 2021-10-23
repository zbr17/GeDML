import yaml
import inspect
from copy import deepcopy

import torch
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP 

from ...config.setting.launcher_setting import (
    DEFAULT_NAME
)

class gradclipper:
    def __init__(self, parameters, max_norm):
        self.parameters = parameters
        self.max_norm = max_norm

    def __call__(self):
        torch.nn.utils.clip_grad_norm_(self.parameters, self.max_norm)

def operate_dict_recursively(src_dict, condition, operation, flag_dict=None, addtion_params=None):
    flag_dict = (
        flag_dict
        if flag_dict is not None
        else src_dict
    )
    dst_dict = dict()
    for k, v in flag_dict.items():
        if condition(k, v):
            dst_dict[k] = operate_dict_recursively(
                src_dict=src_dict[k],
                condition=condition,
                operation=operation,
                flag_dict=flag_dict[k],
                addtion_params=addtion_params
            )
        else:
            dst_dict[k] = operation(v, addtion_params)
    return dst_dict

def load_yaml(yaml_path):
    return yaml.load(
        stream=open(yaml_path, mode='r'),
        Loader=yaml.FullLoader
    )

def get_first_dict_key(data_dict):
    return list(data_dict.keys())[0]

def get_first_dict_value(data_dict):
    return list(data_dict.values())[0]

def get_last_linear(model):
    for name in ['fc', 'last_linear']:
        last_layer = getattr(model, name, None)
        if last_layer is not None:
            return last_layer, name
    raise KeyError("Structure missing: There is no fc or last_linear in the givin model!")

def set_last_linear(model, target):
    setattr(model, get_last_linear(model)[1], target) 

def set_layers_to_eval(layer_name):
    def set_to_eval(m):
        classname = m.__class__.__name__
        if classname.find(layer_name) != -1:
            m.eval()
    return set_to_eval

def func_params_mediator(params_bank, init_func):
    """
    Pass parameters to a function according to func's parameters list (through inspect package)

    Args:
        params_bank (list):
            List of the parameters bank. Each item must be the instance of ``dictionary`` or ``object``.
        init_func (func):
            Function to be called.
    
    Returns:
        The output of the function.
    """
    # get the completed params
    assert isinstance(params_bank, list)

    # check
    signature = inspect.getfullargspec(init_func)
    all_params = signature.args
    params_len = len(all_params)
    optional_len = (
        len(signature.defaults) 
        if signature.defaults is not None
        else 0
    )
    required_params = [item for item in all_params[:(params_len-optional_len)] if item != 'self']

    # construct params_dict
    curr_params = {}
    for params_name in all_params:
        for curr_bank in params_bank:
            if isinstance(curr_bank, dict):
                params_value = curr_bank.get(params_name, None)
            else:
                params_value = getattr(curr_bank, params_name, None)
            
            if params_value is not None:
                curr_params[params_name] = params_value

    assert all([item in list(curr_params.keys()) for item in required_params])
    return init_func(**curr_params)

def get_default(input_object, module_name):
    output = None
    if isinstance(input_object, dict):
        output = input_object.get(module_name, None)
        if output is None:
            output = input_object.get(DEFAULT_NAME, None)
        else:
            output = output.get(DEFAULT_NAME, None)
        return output
    else:
        return input_object

def dict_to_command(input_dict, ):
    command = ""
    for k, v in input_dict.items():
        command += "--{} {} ".format(
            k, v
        )
    return command

"""
Distributed utilities
"""
def distributed_gather_objects(*objects_list, rank=None, world_size=None) -> tuple:
    gathered_objects_list = []
    world_size = (
        dist.get_world_size()
        if world_size is None
        else world_size
    )
    rank = (
        dist.get_rank()
        if rank is None
        else rank
    )
    for objects_item in list(objects_list):
        if objects_item.dim() < 1:
            objects_item = objects_item.unsqueeze(0)

        curr_gather_list = [
            torch.zeros_like(objects_item)
            for _ in range(world_size)
        ]
        dist.all_gather(
            tensor_list=curr_gather_list,
            tensor=objects_item.contiguous()
        )
        curr_gather_list[rank] = objects_item
        curr_gather = torch.cat(curr_gather_list)
        gathered_objects_list.append(curr_gather)
    return tuple(gathered_objects_list)
