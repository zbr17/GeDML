import torch.nn as nn
from torchdistlog import logging
from ...modules import WithRecorder
from ... import models
from ....launcher.misc import utils

# NOTE: These named hyper-parameters are IMPORATANT for transforms!
_hyper_info_ = [
    "input_space", "input_size", "input_range", "mean", "std", "num_classes"
]

class DefaultModelWrapper(WithRecorder):
    def __init__(
        self,
        initiate_method: str,
        base_model: nn.Module,
        *args,
        **kwargs
    ):
        super(DefaultModelWrapper, self).__init__(*args, **kwargs)
        self.initiate_method = initiate_method
        self.base_model = base_model
        self._regester_hyperinfo()
    
    def _regester_hyperinfo(self):
        for _hyper_name in _hyper_info_:
            _hyper_value = getattr(self.base_model, _hyper_name, None)
            if _hyper_value is not None:
                setattr(self, _hyper_name, _hyper_value)
    
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