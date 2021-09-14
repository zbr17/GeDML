import sys
import os
from os.path import join as opj
workspace = os.environ["WORKSPACE"]
sys.path.append(
    opj(workspace, 'code/GeDML/src')
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, nargs='+', default=[2, 3, 4, 5, 6, 7])
parser.add_argument("--batch_size", type=int, default=240)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--num_classes", type=int, default=1000)
parser.add_argument("--dataset", type=str, default='ImageNet')

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
# new
tmux = TmuxManager(
    session_name=tmux_name,
    conda_env="pytorch",
)

# start distributed training
curr_path = os.path.dirname(sys.argv[0])
run_file = os.path.join(
    curr_path,
    "distributed_trigger.py"
)
opt.world_size = torch.cuda.device_count()
for opt.rank, opt.gpu in enumerate(gpu_list):
    tmux.new_window(
        window_name="{}-gpu-{}".format(
            tmux_name, opt.gpu
        )
    )
    cmd = "python {} {}".format(
        run_file,
        utils.dict_to_command(vars(opt))
    )
    print(cmd)
    tmux.send_cmd(cmd)
