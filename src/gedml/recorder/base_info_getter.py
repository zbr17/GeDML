import numpy as np 
import torch
from ..config.setting.recorder_setting import TO_RECORD_LIST
from . import utils

class BaseInfoGetter:
    def __init__(self):
        self.dict_counts = {}
    
    def assert_from_obj(self, obj):
        to_record_list = getattr(obj, TO_RECORD_LIST, None)
        assert to_record_list is not None, 'obj must have attribute: {}!'.format(TO_RECORD_LIST)
        return to_record_list
    
    def convert_from_obj(self, obj, name=None):
        name = obj.__class__.__name__ if name is None else name
        output = {}
        to_record_list = self.assert_from_obj(obj)
        for record_item in to_record_list:
            output[record_item] = utils.get_value(getattr(obj, record_item, 0))
        return {
            self.get_description(name): output
        }

    def get_data(self, data: (dict or ...), name=None) -> (dict, int):
        """
        This is the main API for BaseInfoGetter class.

        Args:
            data        (dict or obj):      Dict or obj (has TO_RECORD_LIST attribute)
            name        (str)
        Returns:
            [0]         (dict):             Data to be recorded
            [1]         (int):              Step
        """
        # extract data to dict
        output = self.convert_from_obj(data, name) if not isinstance(data, dict) else data
        assert len(list(output.keys())) < 2, 'first-level keys must be less than 2 in output dict'
        # increase counts
        primary_key = utils.get_first_key_of_dict(output)
        self.increase_counts(primary_key)
        return output, self.dict_counts[primary_key]
    
    def increase_counts(self, key):
        if key not in self.dict_counts:
            self.dict_counts[key] = 0 
        else:
            self.dict_counts[key] += 1
        
    def get_description(self, name, spliter='-'):
        description = spliter.join([name, 'attribute'])
        return description

    def get_value(self, data):
        if isinstance(data, torch.Tensor):
            return data.item()
        elif isinstance(data, (int, float)):
            return data
        else:
            raise TypeError("Data must be torch.Tensor, int, or float!")
