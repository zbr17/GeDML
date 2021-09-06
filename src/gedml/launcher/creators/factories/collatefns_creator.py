from .base_creator import BaseCreator
from ....core.samplers import collate_fn

class collatefnsCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [collate_fn]
