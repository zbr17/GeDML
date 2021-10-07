from .base_creator import BaseCreator
from torchdistlog import logging
from ....core import transforms
from ...misc import utils

class transformsCreator(BaseCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def prepare_packages(self):
        self.package = [None]
    
    def create(self, module_type, module_args, *args, **kwargs):
        """
        func: create() is specially overrided for instantiating transform objects.
        """
        logging.info("+++ {}'s params: \n{}".format(
            module_type, self.print_params(module_args)
        ))
        # get model_properties
        model = module_args["model_properties"]
        params = module_args["compose_list"]
        try:
            model_properties = {
                k: getattr(model, k) 
                for k in ["mean", "std", "input_space", "input_range"]
            }
        except (KeyError, AttributeError):
            model_properties = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        # get addition_params 
        addition_params = []
        # input_space
        input_space = model_properties.get("input_space", None)
        if input_space == "BGR":
            addition_params.append({
                "ConvertToBGR": {}
            })
        # to tensor
        addition_params.append({
            "ToTensor": {}
        })
        # input_range
        input_range = model_properties.get("input_range", None)
        if isinstance(input_range, list)  and input_range[1] != 1:
            addition_params.append({
                "Multiplier": {
                    "multiple": input_range[1]
                }
            })
        # mean, std
        mean = model_properties.get("mean", None)
        std = model_properties.get("std", None)
        if None not in [mean, std]:
            addition_params.append({
                "Normalize": {
                    "mean": mean,
                    "std": std
                }
            })
        
        # create transform.Compose
        params += addition_params
        compose_list = [
            getattr(
                transforms,
                utils.get_first_dict_key(item),
                None
            )(**utils.get_first_dict_value(item))
            for item in params
        ]
        module_object = transforms.Compose(compose_list)

        # whether to wrap
        wrapper_name = module_args.get("wrapper", False)
        if wrapper_name:
            wrapper = getattr(
                transforms,
                wrapper_name
            )
            module_object = wrapper(module_object)
        return module_object
