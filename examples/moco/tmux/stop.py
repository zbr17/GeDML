import sys
import os
from os.path import join as opj
workspace = os.environ["WORKSPACE"]
sys.path.append(
    opj(workspace, 'code/GeDML/src')
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in opt.gpu])
gpu_list = [gpu for gpu in opt.gpu]

import torch.distributed as dist 
import torch

from gedml.client.tmux import TmuxManager, clear_tmux
from gedml.launcher.misc import utils

# create tmux manager
tmux_name = "borel"
# clear 
clear_tmux(tmux_name)
print("Clear!")
