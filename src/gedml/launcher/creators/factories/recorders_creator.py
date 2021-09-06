from .base_creator import BaseCreator
from ....  import recorder 

class recordersCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [recorder]