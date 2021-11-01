import os
from copy import deepcopy
from torchdistlog import logging
from .creator_manager import CreatorManager
from ..misc import utils
from ...config.setting.launcher_setting import (
    SEARCH_PARAMS_CONDITION,
    SEARCH_PARAMS_FLAG,
    SEARCH_TOP_CLASS,
    SEARCH_TAR_NAME,
    WILDCARD,
    SPLITER,
    ATTRSYM,
    LINK_GENERAL_KEY,
    PIPELINE_KEY,
)

class ConfigHandler:
    """
    This class takes charge of reading yaml files, combining config parameters, modifying local parameters and initializing modules.

    ``ConfigHandler`` will call ``CreatorManager`` to use corresponding sub-creator to do the initialization respectly.

    Args:
        link_path (str):
            Path where the ``link.yaml`` is stored.
        params_path (str):
            Path where the params configs are stored.
        assert_path (str):
            Path where the ``assert.yaml`` is stored. (To be done!)
        is_confirm_first (bool):
            Whether to confirm before starting initialization. default: True.

    Example:
        >>> config_handler = ConfigHandler()
        >>> config_handler.get_params_dict()
        >>> objects_dict = config_handler.create_all()

    Todo:
        - assert parameters.
    """
    def __init__(
        self,
        convert_dict={},
        link_path=None,
        assert_path=None,
        params_path=None,
        is_confirm_first=True,
    ):
        self.convert_dict = convert_dict
        self.link_path = link_path
        self.assert_path = assert_path
        self.params_path = params_path
        self.is_confirm_first = is_confirm_first

        self.creator_manager = CreatorManager()
        self.initiate_params()
    
    @property
    def core_modules_order_to_create(self):
        """
        NOTE: Do NOT arbitrarily change the intiatlization order.
        """
        return [
            "recorders",
            "metrics",
            "models",
            "collectors",
            "selectors",
            "losses",
            "evaluators",
            "optimizers",
            "schedulers",
            "gradclipper",
            "transforms",
            "datasets",
            "samplers",
            "trainers",
            "testers",
            "managers",
        ]
    
    @property
    def output_wrapper_list(self):
        return [
            "models",
            "collectors",
            "selectors",
            "losses"
        ]

    def load_link_config(self):
        self.link_config = utils.load_yaml(self.link_path)
        self.general_setting = self.link_config.pop(LINK_GENERAL_KEY, None)
        self.pipeline_setting = self.link_config.pop(PIPELINE_KEY, None)
    
    def show_link(self):
        logging.info("#####################")
        logging.info("Load link config")
        logging.info("#####################")
        for k, v in self.link_config.items():
            if k != LINK_GENERAL_KEY and k != PIPELINE_KEY:
                logging.info("... {}".format(k))
                assert isinstance(v, list)
                for item in v:
                    logging.info("... ... {}: {}".format(
                            utils.get_first_key(item),
                            utils.get_first_value(item)
                        )
                    )
        if self.is_confirm_first:
            _ = input("Confirm: ...")

    def initiate_params(self):
        self.load_link_config()
        self.assert_dict = utils.load_yaml(self.assert_path)
        self.show_link()
    
    """
    config setting
    """
    def register_packages(self, module_name, extra_package):
        """
        Register new packages into the specific module-creator.

        Args:
            module_name (str):
                The specific module-creator.
            extra_package (list or module):
                Extra packages to be added.
        """
        self.creator_manager.register_packages(module_name, extra_package)
    
    """
    About construction of link config
    """

    def _maybe_modify_link_dict(self, modify_link_dict=None):
        if modify_link_dict is not None:
            for k, v in modify_link_dict.items():
                assert isinstance(v, list)
                for item in v:
                    key = utils.get_first_key(item)
                    value = utils.get_first_value(item)
                    index = -1
                    target_list = self.link_config.get(k, [])
                    for idx, t_item in enumerate(target_list):
                        if utils.get_first_key(t_item) == key:
                            index = idx
                    if index == -1:
                        self.link_config[k] = target_list
                        self.link_config[k].append(
                            {key: value}
                        )
                        logging.info("Add {} to {}".format(key, value))
                    else:
                        self.link_config[k][index] = {key: value}
                        logging.info("Modify {} to {}".format(key, value))
            logging.info("Link-config has been modified!")
        return self.link_config

    def _generate_pipeline_flow_chart(self):
        for src_module, info in self.wrapper_dict.items():
            for src_inst, sub_info in info.items():
                for src_group, (sub_k, _) in enumerate(sub_info["map"].items()):
                    sub_k = sub_k.replace("->", "/")
                    dst_module, dst_inst, dst_tag = sub_k.split("/")
                    # get input-list
                    input_list_str = "\n".join(self.wrapper_dict[src_module][src_inst]["input"])
                    # get output-list
                    output_list_str = "\n".join(list(self.wrapper_dict[src_module][src_inst]["map"][sub_k].keys()))
                    # get next-input-list
                    if self.wrapper_dict.get(dst_module, False):
                        next_input_list_str = "\n".join(self.wrapper_dict[dst_module][dst_inst]["input"])
                    else:
                        next_input_list_str = "null"
                    # form the edge
                    self.pipeline_to_save.append(
                        (
                            "INPUT\n{}\nNAME\n{}/{}".format(input_list_str, src_module, src_inst),
                            "INPUT\n{}\nNAME\n{}/{}".format(next_input_list_str, dst_module, dst_inst),
                            "GROUP-{}\nTAG-{}\n{}".format(src_group, dst_tag, output_list_str),
                        )
                    )
        return self.pipeline_to_save

    """
    About loading and generating params_dict
    """
    
    def _get_params_dict_operation(self):
        """
        Parse params_dict and wrapper_dict recursively.
        """
        for module_type, data_info in self.link_config.items():
            assert isinstance(data_info, list)
            self.params_dict[module_type] = {}
            if module_type in self.output_wrapper_list:
                self.wrapper_dict[module_type] = {}
            for item in data_info:
                # parameters
                item_key = utils.get_first_key(item)
                item_value = utils.get_first_value(item)
                item_params_path = os.path.join(self.params_path, module_type, item_value)
                item_params_dict = utils.load_yaml(item_params_path)
                self.params_dict[module_type][item_key] = item_params_dict
                # wrapper
                if module_type in self.output_wrapper_list:
                    wrapper_info = self.pipeline_setting[module_type][item_key]
                    new_map = {"input": wrapper_info["input"], "map": {}}
                    for sub_k, sub_v in wrapper_info["map"].items():
                        if "->" in sub_k:
                            sub_k = sub_k.replace("->", "/")
                        else:
                            sub_k = sub_k + "/"
                        new_map["map"][sub_k] = {
                            wrapper_info["output"][idx]: idx
                            for idx in sub_v
                        }
                    ### wildcard character SETTING
                    logging.warning("Remain to handle the wildcard characters!") # TODO:
                    self.wrapper_dict[module_type][item_key] = new_map
        return self.params_dict, self.wrapper_dict

    def get_params_dict(self, link_config=None, modify_link_dict=None):
        """
        Read and combine config parameters.

        Args:
            link_config (dict):
                Link dictionary.
            modify_link_dict (dict):
                Modify link config. (Default = None)

        Returns:
            dict: params' dictionary.
        """
        self.link_config = (
            self.link_config
            if link_config is None
            else link_config
        )
        self.params_dict = {}
        self.wrapper_dict = {}
        self.pipeline_to_save = []

        # maybe modify link dict
        self.link_config = self._maybe_modify_link_dict(modify_link_dict)
        # generate params_dict
        self.params_dict, self.wrapper_dict = self._get_params_dict_operation()
        # generate pipeline flow chart 
        self.pipeline_to_save = self._generate_pipeline_flow_chart()
        
        return self.params_dict
    
    """
    About initialization of objects
    """

    def _maybe_search_params(self, module_args, instance_name):
        class_name = utils.get_first_key(module_args)
        module_params = module_args[class_name]
        if isinstance(module_params, dict):
            for k, v in module_params.items():
                search_func_name = SEARCH_PARAMS_CONDITION(v)
                if search_func_name:
                    top_class = SEARCH_TOP_CLASS(v)
                    target_name = SEARCH_TAR_NAME(v)
                    search_func_name = search_func_name.replace(SEARCH_PARAMS_FLAG, "").lower()
                    module_params[k] = getattr(self, search_func_name)(
                        top_class=top_class, 
                        instance_name=instance_name, 
                        target_name=target_name
                    )
        return module_params
    
    def _search_with_same_name_(self, top_class, instance_name, **kwargs):
        instance_dict = utils.operate_dict_recursively(
            src_dict=self.objects_dict[top_class],
            condition=lambda k, v: v==instance_name,
            operation=lambda x, params: x
        )
        return instance_dict[instance_name]
    
    def _search_with_target_name_(self, top_class, target_name, **kwargs):
        instance_dict = utils.operate_dict_recursively(
            src_dict=self.objects_dict[top_class],
            condition=lambda k, v: v==target_name,
            operation=lambda x, params: x
        )
        return instance_dict[target_name]
    
    def _search_with_target_attr_(self, top_class, target_name, **kwargs):
        module_name, attr_name = target_name.split("/")
        instance_dict = utils.operate_dict_recursively(
            src_dict=self.objects_dict[top_class],
            condition=lambda k, v: v==module_name,
            operation=lambda x, params: x
        )
        return getattr(instance_dict[module_name], attr_name)
    
    def _pass_with_named_member_(self, target_name, **kwargs):
        return getattr(
            self, target_name
        )

    def get_objects_dict(self, params_dict=None):
        self.objects_dict = {}
        params_dict = self.params_dict if params_dict is None else params_dict
        for k in self.core_modules_order_to_create:
            v = params_dict.get(k, False)
            if v:
                assert isinstance(v, dict) 
                output = {}
                for sub_k, sub_v in v.items():
                    self._maybe_search_params(
                        module_args=sub_v,
                        instance_name=sub_k
                    )
                    output[sub_k] = self.creator_manager.create(
                        module_type=k,
                        module_params=sub_v,
                    )
                    logging.info(
                        "... {}: {} created, id={}".format(
                            sub_k,
                            utils.get_first_key(sub_v),
                            id(output[sub_k])
                        )
                    )
                    logging.info("---------------------------")
                self.objects_dict[k] = output
        return self.objects_dict

    def maybe_modify_params_dict(self, params_dict, change_dict={}):
        assert isinstance(change_dict, dict)
        for k, v in change_dict.items():
            modify_list = self.convert_dict[k]
            for modify_path in modify_list:
                top_class, remain_str = modify_path.split(SPLITER)
                instance_name, attr_name = remain_str.split(ATTRSYM)
                # search the instance
                instance_dict = utils.operate_dict_recursively(
                    src_dict=params_dict[top_class],
                    condition=lambda k, v: v==instance_name,
                    operation=lambda x, params: x
                ).get(instance_name, None)
                # change parameters
                if instance_dict is None:
                    logging.warning("{}/{} doesn't exist! modify failed!".format(
                        top_class, instance_name
                    ))
                    raise KeyError("Plase check key/value pairs!")
                else:
                    class_name = utils.get_first_key(instance_dict)
                    instance_dict[class_name][attr_name] = v
                    logging.info("{}/{}/{} is changed to {}".format(
                        top_class, instance_name, attr_name, v
                    ))

        return params_dict
    
    def create_all(self, change_dict={}):
        """
        Initialize all modules according to params dictionary. 
        Redundant options will be popped from the input change_dict.

        Args:
            change_dict (dict):
                Dictionary that overwrites certain parameters. (optional)
        
        Returns:
            dict: initialized objects dictionary.
        """
        # Pop redundant options
        change_dict = deepcopy(change_dict)
        convert_opt_list = list(self.convert_dict.keys())
        for k in list(change_dict.keys()):
            if k not in convert_opt_list:
                logging.info("'{}' has been popped from 'change-dict' by config-handler".format(k))
                change_dict.pop(k)

        logging.info("#####################")
        logging.info("Create objects")
        logging.info("#####################")

        # update according to argparse
        self.params_dict = self.maybe_modify_params_dict(self.params_dict, change_dict)

        # update according to general_setting in link_config
        if self.general_setting is not None:
            self.params_dict = self.maybe_modify_params_dict(
                self.params_dict, self.general_setting
            )

        self.maybe_assert_params_dict(self.params_dict)
        self.objects_dict = self.get_objects_dict()        
        return self.objects_dict

    def maybe_assert_params_dict(self, params_dict):
        logging.warning("'maybe_assert_params_dict' is not implemented!")

    """
    Other functions
    """

    def get_certain_params_dict(self, change_list, params_dict=None): # TODO: 
        params_dict = (
            self.params_dict
            if params_dict is None
            else params_dict
        )
        assert isinstance(change_list, list)
        output_dict = {}
        for k in change_list:
            curr_dict = {}
            modify_list = self.convert_dict[k]
            for modify_path in modify_list:
                top_class, remain_str = modify_path.split(SPLITER)
                instance_name, attr_name = remain_str.split(ATTRSYM)
                # search the instance
                instance_dict = utils.operate_dict_recursively(
                    src_dict=params_dict[top_class],
                    condition=lambda k, v: v==instance_name,
                    operation=lambda x, params: x
                )[instance_name]
                # record parameters
                curr_dict[modify_path] = instance_dict[attr_name]
            output_dict[k] = curr_dict
        return output_dict
