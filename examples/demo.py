import os
import sys
import random
from os.path import join as opj
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
workspace = os.environ["WORKSPACE"]
sys.path.append(
    opj(workspace, 'code/GeDML/src')
)

from gedml.launcher.misc import ParserWithConvert
from gedml.launcher.creators import ConfigHandler
from gedml.launcher.misc import utils
from torchdistlog import logging
logging.getLogger().setLevel(logging.INFO)

# argparser
csv_path = os.path.abspath(opj(__file__, "../config/args.csv"))
parser = ParserWithConvert(csv_path=csv_path, name="GeDML")
opt, convert_dict = parser.render()

# args postprocess
opt.save_path = opj(opt.save_path, opt.save_name)
if opt.is_resume:
    opt.delete_old = False

# hyper-parameters
start_epoch = 0
phase = "train"
is_test = True
is_save = True
warm_up = opt.warm_up
warm_up_list = opt.warm_up_list
is_distributed = opt.is_distributed
seed = opt.seed

logging.set_dist(is_dist=is_distributed)
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = True

# get confighandler
config_root = os.path.abspath(opj(__file__, "../config/"))
if opt.link_path is None:
    link_root = os.path.join(config_root, "links")
    if opt.setting is None:
        opt.link_path = opj(link_root, "link.yaml")
    else:
        opt.link_path = os.path.join(link_root, "link_" + opt.setting + ".yaml")
opt.assert_path = os.path.join(config_root, "assert.yaml")
opt.param_path = os.path.join(config_root, "param")
opt.wrapper_path = os.path.join(config_root, "wrapper")

config_handler = ConfigHandler(
    convert_dict=convert_dict,
    link_path=opt.link_path,
    assert_path=opt.assert_path,
    params_path=opt.param_path,
    wrapper_path=opt.wrapper_path,
    is_confirm_first=True
)

# initiate params_dict
params_dict = config_handler.get_params_dict(
    modify_link_dict={
        "datasets": [
            {"train": "{}_train.yaml".format(opt.dataset)},
            {"test": "{}_test.yaml".format(opt.dataset)}
        ]
    }
)

# delete redundant options
opt_dict = deepcopy(opt.__dict__)
convert_opt_list = list(config_handler.convert_dict.keys())
for k in list(opt_dict.keys()):
    if k not in convert_opt_list:
        opt_dict.pop(k)

# modify parameters
objects_dict = config_handler.create_all(opt_dict)

# get manager
manager = utils.get_default(objects_dict, "managers")

# get recorder
recorder = utils.get_default(objects_dict, "recorders")

# start
manager.run(
    phase=phase,
    start_epoch=start_epoch,
    is_test=is_test,
    is_save=is_save,
    warm_up=warm_up,
    warm_up_list=warm_up_list
)


