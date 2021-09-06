from .base_creator import BaseCreator
from ....core import evaluators

class evaluatorsCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [evaluators]