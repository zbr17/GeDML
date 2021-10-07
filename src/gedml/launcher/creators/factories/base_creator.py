from abc import ABCMeta, abstractmethod
from ....config.setting.launcher_setting import (
    CLASS_KEY,
    PARAMS_KEY,
    INITIATE_KEY,
)
from torchdistlog import logging
import torch

class BaseCreator(metaclass=ABCMeta):
    def __init__(self, creator_mode=None):
        self.creator_mode = creator_mode
        self.init_packages()

    def __call__(self, module_params):
        assert isinstance(module_params, dict) 
        module_type = module_params[CLASS_KEY]
        module_args = module_params[PARAMS_KEY]
        self.creator_mode = module_params[INITIATE_KEY]
        module_args = {} if module_args is None else module_args
        return self.create(module_type, module_args)
    
    def init_packages(self):
        self.prepare_packages()
        assert isinstance(self.package, list)
    
    def register_packages(self, extra_package):
        """
        Register new packages.

        Args:
            extra_package (list or module):
                Extra packages to be registered.
        """
        if isinstance(extra_package, list):
            self.package = extra_package + self.package
        else:
            self.package = [extra_package] + self.package
        logging.info("{} has registered new packages {}".format(
            self.__class__.__name__, extra_package
        ))

    @abstractmethod
    def prepare_packages(self):
        self.package = None
    
    def create(self, module_type, module_args, *args, **kwargs):
        logging.info("+++ {}'s params: \n{}".format(
            module_type, self.print_params(module_args)
        ))
        # get class
        module_class = self.get_class(module_type)
        # create object
        return self.create_object(module_class, module_args)
    
    def print_params(self, args):
        args_string = ""
        for k, v in args.items():
            if isinstance(v, torch.nn.Module):
                v = "module-" + v.__class__.__name__
            elif isinstance(v, dict):
                v = "dict-" + ";".join(v.keys())
            args_string += "\t\t\t{}: {}\n".format(k, v)
        return args_string
    
    def get_class(self, module_type):
        module_class = None
        for sub_package in self.package:
            module_class = getattr(sub_package, module_type, None)
            if module_class is not None:
                break
        assert module_class is not None, "Error Info: {} - {}".format(
            sub_package, module_type
        )
        return module_class
    
    def create_object(self, module_class, module_args):
        module_args = self.maybe_modify_args(module_args)
        module_object = module_class(**module_args)
        module_object = self.maybe_modify_object(module_object)
        return module_object
    
    def maybe_modify_args(self, module_args):
        return module_args

    def maybe_modify_object(self, module_object):
        return module_object