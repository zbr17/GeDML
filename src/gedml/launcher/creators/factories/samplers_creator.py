from .base_creator import BaseCreator
from ....core import samplers

class samplersCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [samplers]
    
    def maybe_modify_args(self, module_args):
        module_args["labels"] = module_args["labels"].get_labels()
        return module_args