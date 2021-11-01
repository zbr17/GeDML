import os

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