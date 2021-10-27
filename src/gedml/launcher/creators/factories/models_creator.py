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
        # if wrapping
        self.wrapper_name = module_args.pop("WRAPPER", None)
        self.wrapper_params = {}
        for key in list(module_args.keys()):
            if "WRAPPER_" in key:
                self.wrapper_params[key.replace("WRAPPER_", "")] = module_args.pop(key)
        return module_args
    
    def maybe_modify_object(self, module_object: torch.nn.Module):
        if self.creator_mode is None:
            pass
        elif self.creator_mode == "delete_last_linear":
            if not isinstance(module_object, models.MLP):
                utils.set_last_linear(module_object, models.Identity())
            logging.info("Creator MODE: Delete last linear: {}".format(module_object.__class__.__name__))
        elif self.creator_mode == "freeze_all":
            if not isinstance(module_object, models.MLP):
                utils.set_last_linear(module_object, models.Identity())
            for param in module_object.parameters():
                param.requires_grad = False
            logging.info("Creator MODE: freeze all: {}".format(module_object.__class__.__name__))
        elif self.creator_mode == "freeze_except_last_linear":
            if not isinstance(module_object, models.MLP):
                utils.set_last_linear(module_object, models.Identity())
            for name, param in module_object.named_parameters():
                if name not in []:
                    param.requires_grad = False
            logging.info("Creator MODE: freeze except last linear: {}".format(module_object.__class__.__name__))
        else:
            raise KeyError("Invalid creator_type: {}, in {}".format(
                self.creator_mode, self.__class__.__name__
            ))
        
        # whether to wrap
        if self.wrapper_name is not None:
            wrapper = getattr(
                models,
                self.wrapper_name
            )
            module_object = wrapper(
                module_object,
                **self.wrapper_params
            )
        return module_object