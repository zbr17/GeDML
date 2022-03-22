import torch
from torchdistlog import logging
from .base_creator import BaseCreator
from ....core import models
from ...misc import utils

class modelsCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [models]
    
    def maybe_modify_args(self, module_args):
        # if initializing
        self.initiate_name = module_args.pop("INITIATE", None)
        # if wrapping
        self.wrapper_args = module_args.pop("WRAPPER", None)
        return module_args
    
    def maybe_modify_object(self, module_object):
        # get wrapper
        if self.wrapper_args is not None:
            wrapper_name = utils.get_first_key(self.wrapper_args)
            wrapper_params = self.wrapper_args[wrapper_name]
            wrapper = getattr(models, wrapper_name)
            module_object = wrapper(
                initiate_method=self.initiate_name,
                base_model=module_object,
                **wrapper_params
            )
        else:
            module_object = models.OneStream(
                initiate_method=self.initiate_name,
                base_model=module_object,
            )
        return module_object
    