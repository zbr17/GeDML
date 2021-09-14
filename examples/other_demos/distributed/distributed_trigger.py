import sys
import os 
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
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--dataset", type=str, default='ImageNet')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu)
    opt.gpu = 0

    runner = DistributedRunner(
        opt
    )
    runner.run()

if __name__ == '__main__':
    subprocess_start()