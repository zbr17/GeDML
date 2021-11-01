import torch.nn as nn
from torchdistlog import logging
from ...modules import WithRecorder
from ... import models
from ....launcher.misc import utils

class DefaultModelWrapper(WithRecorder):
    def __init__(
        self,
        initiate_method: str,
        base_class,
        base_args,
        *args,
        **kwargs
    ):
        super(DefaultModelWrapper, self).__init__(*args, **kwargs)
        self.initiate_method = initiate_method
        self.base_class = base_class
        self.base_args = base_args
    
    def _initiate_model_(self, model, initiate_method):
        # if initializing
        if initiate_method is None:
            pass
        elif initiate_method == "delete_last_linear":
            if not isinstance(model, models.MLP):
                utils.set_last_linear(model, models.Identity())
            logging.info("Creator MODE: Delete last linear: {}".format(model.__class__.__name__))
        elif initiate_method == "freeze_all":
            if not isinstance(model, models.MLP):
                utils.set_last_linear(model, models.Identity())
            for param in model.parameters():
                param.requires_grad = False
            logging.info("Creator MODE: freeze all: {}".format(model.__class__.__name__))
        elif initiate_method == "freeze_except_last_linear":
            if not isinstance(model, models.MLP):
                utils.set_last_linear(model, models.Identity())
            for name, param in model.named_parameters():
                if name not in []:
                    param.requires_grad = False
            logging.info("Creator MODE: freeze except last linear: {}".format(model.__class__.__name__))
        else:
            raise KeyError("Invalid creator_type: {}, in {}".format(
                initiate_method, self.__class__.__name__
            ))