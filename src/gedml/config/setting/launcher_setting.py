import os

CLASS_KEY = "type"
PARAMS_KEY = "params"
INITIATE_KEY = "INITIATE"
WRAPPER_KEY = "WRAPPER"
OBJECTS_KEY = "objects"
DEFAULT_NAME = "default"
WILDCARD = "*"
SPLITER = "/"
ATTRSYM = "-"

LINK_GENERAL_KEY = "LINK_SETTING"
PIPELINE_KEY = "PIPELINE_SETTING"

def search_params_condition(params):
    if isinstance(params, list):
        if len(params) > 0:
            if isinstance(params[0], str):
                if SEARCH_PARAMS_FLAG in params[0]:
                    return params[0]
    return False

SEARCH_PARAMS_FLAG = "~~"
SEARCH_PARAMS_CONDITION = search_params_condition
SEARCH_TOP_CLASS = lambda x: x[1]
SEARCH_TAR_NAME = lambda x: x[2]

INITATE_ORDER_LIST = [
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

WRAPPER_LIST = [
    "models",
    "collectors",
    "selectors",
    "losses"
]