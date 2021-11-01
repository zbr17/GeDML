
import torch.nn as nn
from .default_model_wrapper import DefaultModelWrapper

class OneStream(DefaultModelWrapper):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super(OneStream, self).__init__(*args, **kwargs)

        self.initiate_model()
    
    def initiate_model(self):
        # TODO
        self.base_model = self.base_class(**self.base_args)
        self._initiate_model_(self.base_model, self.initiate_method)
    
    def forward(self, data):
        return self.base_model(data) # To debug