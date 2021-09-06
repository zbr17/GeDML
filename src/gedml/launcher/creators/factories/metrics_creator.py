from .base_creator import BaseCreator

class metricsCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_packages(self):
        from ....core import metrics
        self.package = [metrics]