import torch
from collections import defaultdict

from . import utils

class Storage:    
    def __init__(self, wrapper_params):
        self.indices_dict = {}
        self.wrapper_params = wrapper_params

    def get(self, item_name):
        return getattr(self, item_name)
    
    def get_list(self, item_name_list):
        return [
            getattr(self, item_name)
            for item_name in item_name_list
        ]
    
    def get_dict(self, item_name_list):
        return {
            item_name: getattr(self, item_name, None) # NOTE: if not exist, return **None**
            for item_name in item_name_list
        }
    
    def filter_dict(self, item_name_list, total_dict):
        output_dict = {}
        for item_name in item_name_list:
            item_value = total_dict.get(item_name, None)
            if item_value is None:
                item_value = getattr(self, item_name, None)
            output_dict[item_name] = item_value
        return output_dict

    def add(self, item_name, value):
        setattr(self, item_name, value)
    
    # def add_list(self, item_name_list: list, value_list: list):
    #     if len(item_name_list) == 1:
    #         setattr(self, item_name_list[0], value_list)
    #     else:
    #         for item_name, value in zip(item_name_list, value_list):
    #             setattr(self, item_name, value)
        
    def add_dict(self, item_dict: dict):
        for k in item_dict.keys():
            module, name, group = k.split("/")
            if self.indices_dict.get(module, None):
                if self.indices_dict[module].get(name, None):
                    if self.indices_dict[module][name].get(group, None):
                        self.indices_dict[module][name][group].update(item_dict[k])
                    else:
                        self.indices_dict[module][name][group] = item_dict[k]
                else:
                    self.indices_dict[module][name] = {
                        group: item_dict[k]
                    }
            else:
                self.indices_dict[module] = {
                    name: {group: item_dict[k]}
                }

    def tensors_to_device(self, item_name_list, device):
        for k in item_name_list:
            setattr(
                self,
                k,
                getattr(self, k).to(device)
            )
    
    def output_wrapper(self, sub_value_tuple, sub_k, sub_v, old_k):
        sub_item_output_dict = {
            wrapper_k: sub_value_tuple[wrapper_idx]
            for wrapper_k, wrapper_idx in sub_v.items()
        }
        module_type, name, group = sub_k.split('/')
        if name == "":
            name = old_k
            sub_k = '/'.join([module_type, name, group])
        return sub_item_output_dict, sub_k

    def update(self, modules, cur_module=None):
        """
        For tuple input: wrapper will convert tuple into dict
        For dict input: wrapper will convert dict into dict recursively
        """
        value_dict = defaultdict(dict)

        cur_module_dict = self.indices_dict.get(cur_module, {})
        cur_wrapper_dict = self.wrapper_params.get(cur_module, {})
        if cur_module_dict is not None:
            dict_len = len(cur_module_dict)
            for k, v in cur_module_dict.items():
                # get module
                if not isinstance(modules, dict):
                    module = modules
                    assert dict_len == 1 # if not dict, dict_len must be ONE!
                else:
                    if len(modules) == 1:
                        module = utils.get_first_dict_value(modules)
                    else:
                        module = modules[k]
                
                # pass parameters
                sub_wrapper_dict = cur_wrapper_dict[k]
                for group_key, params in v.items():
                    # get output tuple
                    sub_value_tuple = module( 
                        **self.filter_dict(
                            sub_wrapper_dict["input"],
                            total_dict=params
                        )
                    )
                    # check the type of output tuple and wrap the output
                    if isinstance(sub_value_tuple, dict):
                        for tuple_k, tuple_v in sub_value_tuple.items():
                            for sub_k, sub_v in sub_wrapper_dict["map"].items():
                                sub_item_output_dict, new_sub_k = self.output_wrapper(tuple_v, sub_k, sub_v, k)
                                value_dict[new_sub_k+"{}".format(tuple_k)+group_key].update(sub_item_output_dict)
                    else:
                        if not isinstance(sub_value_tuple, tuple):
                            sub_value_tuple = (sub_value_tuple,)
                        # output wrapper
                        for sub_k, sub_v in sub_wrapper_dict["map"].items():
                            sub_item_output_dict, new_sub_k = self.output_wrapper(sub_value_tuple, sub_k, sub_v, k)
                            value_dict[new_sub_k+group_key].update(sub_item_output_dict)

        # add to storage
        self.add_dict(
            value_dict
        )

        return value_dict
    
    def return_loss_dict(self):
        return self.indices_dict["FINISH"]

    def __repr__(self):
        output_str = "{}\n".format(self.__class__.__name__)
        for k, v in self.__dict__.items():
            output_str += "{}:\n{}\n".format(
                k, str(v)
            )
        return output_str

if __name__ == '__main__':
    test = Storage()
    device = torch.device("cuda:0")
    test.d0 = torch.randn(3, 4)
    test.d1 = torch.randn(3, 4)
    test.tensors_to_device(["d0", "d1"], device)

    pass
