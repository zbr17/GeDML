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
    
    def create_object(self, module_class, module_args):
        # if initializing
        initiate_name = module_args.pop("INITIATE", None)
        # if wrapping
        wrapper_args = module_args.pop("WRAPPER", None)
        if wrapper_args is not None:
            wrapper_name = utils.get_first_key(wrapper_args)
            wrapper_params = wrapper_args[wrapper_name]
            wrapper = getattr(models, self.wrapper_name)
            module_object = wrapper(
                initiate_method=initiate_name,
                base_class=module_class,
                base_args=module_args,
                **wrapper_params
            )
        else:
            module_object = models.OneStream(
                initiate_method=initiate_name,
                base_class=module_class,
                base_args=module_args,
            )
            pass

        return module_object

    