from .base_creator import BaseCreator
from ....core import collectors

class collectorsCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [collectors]
    