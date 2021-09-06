import sys
import os
from os.path import join as opj
workspace = os.environ["WORKSPACE"]
sys.path.append(
    opj(workspace, 'code/GeDML/src')
)
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)

import torch.distributed as dist 

from gedml.launcher.runners.distributed_runner import DistributedRunner

def subprocess_start():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default='ImageNet')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu)
    opt.gpu = 0
    opt.link_path = os.path.join(workspace, "code/Experiments/GeDML/demo/moco/link.yaml")

    runner = DistributedRunner(
        opt
    )
    runner.run()

if __name__ == '__main__':
    subprocess_start()