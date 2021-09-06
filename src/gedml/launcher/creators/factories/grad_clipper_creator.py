from .base_creator import BaseCreator
from ...misc import utils

class gradclipperCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [utils]
    
    def maybe_modify_args(self, module_args):
        module_args['parameters'] = module_args['parameters'].parameters()
        return module_args