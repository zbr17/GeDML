
import torch.nn as nn
from .default_model_wrapper import DefaultModelWrapper

class OneStream(DefaultModelWrapper):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(OneStream, self).__init__(*args, **kwargs)
        self._initiate_model_(self.base_model, self.initiate_method)
    
    def forward(self, data):
        return self.base_model(data)