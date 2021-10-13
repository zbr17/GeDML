import os
from copy import deepcopy
from torchdistlog import logging
from .creator_manager import CreatorManager
from ..misc import utils
from ...config.setting.launcher_setting import (
    CLASS_KEY,
    PARAMS_KEY,
    INITIATE_KEY,
    WRAPPER_KEY,
    OBJECTS_KEY,
    SEARCH_PARAMS_CONDITION,
    SEARCH_PARAMS_FLAG,
    SEARCH_TOP_CLASS,
    SEARCH_TAR_NAME,
    WILDCARD,
    SPLITER,
    ATTRSYM,
    INITATE_ORDER_LIST,
    WRAPPER_LIST,
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
        wrapper_path=None,
        is_confirm_first=True,
    ):
        self.convert_dict = convert_dict
        self.link_path = link_path
        self.assert_path = assert_path
        self.params_path = params_path
        self.wrapper_path = wrapper_path
        self.is_confirm_first = is_confirm_first

        self.creator_manager = CreatorManager()
        self.initiate_params()
    
    @property
    def core_modules_order_to_create(self):
        return INITATE_ORDER_LIST
    
    @property
    def output_wrapper_list(self):
        return WRAPPER_LIST

    def _get_params_dict_operation(self, data_info, module_type):
        """
        Parse params_dict and wrapper_dict recursively.
        """
        assert isinstance(data_info, list)
        self.params_dict[module_type] = {}
        if module_type in self.output_wrapper_list:
            self.wrapper_dict[module_type] = {}
        for item in data_info:
            # parameters
            item_key = utils.get_first_dict_key(item)
            item_value = utils.get_first_dict_value(item)
            item_params_path = os.path.join(self.params_path, module_type, item_value)
            item_params_dict = utils.load_yaml(item_params_path)
            class_value = utils.get_first_dict_key(item_params_dict)
            params_value = item_params_dict[class_value][PARAMS_KEY]
            initiate_value = item_params_dict[class_value].get(INITIATE_KEY, None)
            self.params_dict[module_type][item_key] = {
                CLASS_KEY: class_value,
                PARAMS_KEY: params_value,
                INITIATE_KEY: initiate_value
            }
            # wrapper
            if module_type in self.output_wrapper_list:
                item_wrapper_path = os.path.join(self.wrapper_path, module_type, class_value + ".yaml")
                try:
                    item_wrapper_dict = utils.load_yaml(item_wrapper_path)
                    logging.info("Load specific wrapper config: {}".format(item_wrapper_path))
                except:
                    item_wrapper_path = os.path.join(self.wrapper_path, module_type, "_DEFAULT.yaml")
                    item_wrapper_dict = utils.load_yaml(item_wrapper_path)
                    logging.info("Load default wrapper config: {}".format(item_wrapper_path))
                # search pipeline
                new_map = {}
                for k, v in item_wrapper_dict["map"].items():
                    query_key = "/".join([module_type, item_key, k])
                    target_route = None
                    for pipe_k, pipe_v in self.pipeline_setting.items():
                        # TODO: for multi output 
                        # for single output
                        if query_key in pipe_k:
                            target_route = pipe_v
                    assert target_route is not None, "NO target-route matched with {}".format(query_key)
                    new_k = target_route
                    new_map[new_k] = v
                item_wrapper_dict["map"] = new_map
                self.wrapper_dict[module_type][item_key] = item_wrapper_dict
    
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
    About construction of objects dict
    """

    def _get_objects_dict_operation(self, params_info, module_type):
        assert isinstance(params_info, dict) 
        output = {}
        for k, v in params_info.items():
            self._maybe_search_params(
                module_params=v,
                instance_name=k
            )
            output[k] = self.creator_manager.create(
                module_type=module_type,
                module_params=v,
            )
            logging.info(
                "... {}: {} created, id={}".format(
                    k,
                    v[CLASS_KEY],
                    id(output[k])
                )
            )
        return output
    
    def _maybe_search_params(self, module_params, instance_name):
        module_args = module_params[PARAMS_KEY]
        if isinstance(module_args, dict):
            for k, v in module_args.items():
                search_func_name = SEARCH_PARAMS_CONDITION(v)
                if search_func_name:
                    top_class = SEARCH_TOP_CLASS(v)
                    target_name = SEARCH_TAR_NAME(v)
                    search_func_name = search_func_name.replace(SEARCH_PARAMS_FLAG, "").lower()
                    module_args[k] = getattr(self, search_func_name)(
                        top_class=top_class, 
                        instance_name=instance_name, 
                        target_name=target_name
                    )
        return module_args
    
    def _search_with_same_name_(self, top_class, instance_name, target_name):
        instance_dict = utils.operate_dict_recursively(
            src_dict=self.objects_dict[top_class],
            condition=lambda k, v: v==instance_name,
            operation=lambda x, params: x
        )
        return instance_dict[instance_name]
    
    def _search_with_target_name_(self, top_class, instance_name, target_name):
        instance_dict = utils.operate_dict_recursively(
            src_dict=self.objects_dict[top_class],
            condition=lambda k, v: v==target_name,
            operation=lambda x, params: x
        )
        return instance_dict[target_name]
    
    def _search_with_target_attr_(self, top_class, instance_name, target_name):
        module_name, attr_name = target_name.split("/")
        instance_dict = utils.operate_dict_recursively(
            src_dict=self.objects_dict[top_class],
            condition=lambda k, v: v==module_name,
            operation=lambda x, params: x
        )
        return getattr(instance_dict[module_name], attr_name)
    
    def _pass_with_named_member_(self, top_class, instance_name, target_name):
        return getattr(
            self, target_name
        )
    
    """
    About construction of link config and params dict
    """
    
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
        link_config = (
            self.link_config
            if link_config is None
            else link_config
        )

        # maybe modify link dict
        if modify_link_dict is not None:
            for k, v in modify_link_dict.items():
                assert isinstance(v, list)
                for item in v:
                    key = utils.get_first_dict_key(item)
                    value = utils.get_first_dict_value(item)
                    index = -1
                    target_list = link_config.get(k, [])
                    for idx, t_item in enumerate(target_list):
                        if utils.get_first_dict_key(t_item) == key:
                            index = idx
                    if index == -1:
                        link_config[k] = target_list
                        link_config[k].append(
                            {key: value}
                        )
                        logging.info("Add {} to {}".format(key, value))
                    else:
                        link_config[k][index] = {key: value}
                        logging.info("Modify {} to {}".format(key, value))
            logging.info("Link-config has been modified!")
        
        self.params_dict = {}
        self.wrapper_dict = {}
        self.pipeline_to_save = []
        for k, v in link_config.items():
            self._get_params_dict_operation(v, k)
        
        # generate pipeline flow chart
        for k, v in self.pipeline_setting.items():
            src_module, src_inst, src_group = k.split("/")
            dst_module, dst_inst, dst_tag = v.split("/")
            # get input-list
            input_list_str = "\n".join(self.wrapper_dict[src_module][src_inst]["input"])
            # get output-list
            output_list_str = "\n".join(list(self.wrapper_dict[src_module][src_inst]["map"][v].keys()))
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
        
        return self.params_dict
    
    def get_objects_dict(self, params_dict, link_config):
        self.objects_dict = {}
        condition = lambda k, v: isinstance(v, dict)
        for k in self.core_modules_order_to_create:
            if link_config.get(k, False):
                if condition(k, link_config[k]):
                    self.objects_dict[k] = utils.operate_dict_recursively(
                        src_dict=params_dict[k],
                        condition=condition,
                        operation=self._get_objects_dict_operation,
                        flag_dict=link_config[k],
                        addtion_params=k
                    )
                else:
                    self.objects_dict[k] = self._get_objects_dict_operation(params_dict[k], k)
        return self.objects_dict
    
    def load_link_config(self):
        self.link_config = utils.load_yaml(self.link_path)
        self.general_setting = self.link_config.pop(LINK_GENERAL_KEY, None)
        self.pipeline_setting = self.link_config.pop(PIPELINE_KEY, None)
        self._process_pipeline()
    
    def _process_pipeline(self):
        self.pipeline_setting = [
            [sub_item.strip() for sub_item in item.split("->")] 
            for item in self.pipeline_setting
        ]
        self.pipeline_setting = {
            item[0]: item[1]
            for item in self.pipeline_setting
        }

    def initiate_params(self):
        self.load_link_config()
        self.assert_dict = utils.load_yaml(self.assert_path)
        self.show_link()
    
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
                            utils.get_first_dict_key(item),
                            utils.get_first_dict_value(item)
                        )
                    )
        if self.is_confirm_first:
            _ = input("Confirm: ...")

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
        self.objects_dict = self.get_objects_dict(self.params_dict, self.link_config)        
        return self.objects_dict
    
    def get_certain_params_dict(self, change_list, params_dict=None):
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
                curr_dict[modify_path] = instance_dict[PARAMS_KEY][attr_name]
            output_dict[k] = curr_dict
        return output_dict

    
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
                    instance_dict[PARAMS_KEY][attr_name] = v
                    logging.info("{}/{}/{} is changed to {}".format(
                        top_class, instance_name, attr_name, v
                    ))

        return params_dict

    def maybe_assert_params_dict(self, params_dict):
        logging.warning("'maybe_assert_params_dict' is not implemented!")
