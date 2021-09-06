from .base_creator import BaseCreator
from torch.optim import lr_scheduler

class schedulersCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [lr_scheduler]