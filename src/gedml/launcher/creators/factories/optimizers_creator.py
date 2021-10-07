from .base_creator import BaseCreator
from torch import optim
import torch
from torchdistlog import logging

class optimizersCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [optim]
    
    def maybe_modify_args(self, module_args):
        if isinstance(module_args["params"], torch.nn.Module):
            parameters = list(filter(lambda p: p.requires_grad, module_args['params'].parameters()))
            logging.info(
                "Optimizer params from: {} id: {}".format(
                    module_args["params"].__class__.__name__,
                    id(module_args["params"])
                )
            )
        else:
            parameters = module_args["params"]
            logging.info(
                "Optimizer params from: {} id: {}".format(
                    str(parameters), id(parameters)
                )
            )
        module_args['params'] = parameters
        return module_args