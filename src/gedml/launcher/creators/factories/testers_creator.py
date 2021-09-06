from .base_creator import BaseCreator
from ... import testers

class testersCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [testers]
    