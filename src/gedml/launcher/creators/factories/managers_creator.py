from .base_creator import BaseCreator
from ... import managers
from ...misc import utils

class managersCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [managers]
