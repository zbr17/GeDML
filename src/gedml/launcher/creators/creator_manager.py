from .factories.metrics_creator import metricsCreator
from .factories.collectors_creator import collectorsCreator
from .factories.selectors_creator import selectorsCreator
from .factories.losses_creator import lossesCreator
from .factories.models_creator import modelsCreator
from .factories.evaluators_creator import evaluatorsCreator
from .factories.optimizers_creator import optimizersCreator
from .factories.schedulers_creator import schedulersCreator
from .factories.grad_clipper_creator import gradclipperCreator
from .factories.transforms_creator import transformsCreator
from .factories.datasets_creator import datasetsCreator
from .factories.samplers_creator import samplersCreator
from .factories.recorders_creator import recordersCreator
from .factories.managers_creator import managersCreator
from .factories.trainers_creator import trainersCreator
from .factories.testers_creator import testersCreator

class CreatorManager:
    def __init__(self):
        self.registered_creators = [
            "metricsCreator",
            "collectorsCreator",
            "selectorsCreator",
            "lossesCreator",
            "modelsCreator",
            "evaluatorsCreator",
            "optimizersCreator",
            "schedulersCreator",
            "gradclipperCreator",
            "transformsCreator",
            "datasetsCreator",
            "samplersCreator",
            "recordersCreator",
            "managersCreator",
            "trainersCreator",
            "testersCreator"
        ]
        self.creators_dict = {
            name: eval(name)()
            for name in self.registered_creators
        }
    
    def register_packages(self, module_name, extra_package):
        """
        Register new packages into the specific module-creator.

        Args:
            module_name (str):
                The specific module-creator.
            extra_package (list or module):
                Extra packages to be added.
        """
        creator_name = module_name + 'Creator'
        creator_item = self.creators_dict[creator_name]
        creator_item.register_packages(extra_package)

    def create(self, module_type, module_params):
        creator_name = module_type + 'Creator'
        creator_item = self.creators_dict[creator_name]
        creator_output = creator_item(module_params)
        return creator_output