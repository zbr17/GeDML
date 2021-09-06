from .base_creator import BaseCreator
from ... import trainers

class trainersCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [trainers]
