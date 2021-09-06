"""
Depreciated: last update 2021-3
"""

import sys
import logging
import yaml
import os
logging.getLogger().setLevel(logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 5, 6, 7"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "10002"

import torch
import torch.multiprocessing as mp 
import torch.distributed as dist 
import torchvision.transforms as transforms
torch.backends.cudnn.deterministic = True

from gedml.launcher.creators import ConfigHandler
from gedml.launcher.misc import utils

# TODO: to be perfected

def main_worker(gpu, world_size, batch_size, num_workers):
    # initiate torch.distributed
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=gpu
    )

    # get config_handler
    config_handler = ConfigHandler()

    # initiate params_dict
    config_handler.get_params_dict()

    # modify parameters: save_path, gpu_device
    save_path = config_handler.get_certain_params_dict(["save_path"])
    save_path = utils.get_first_dict_value(save_path["save_path"])
    save_path = os.path.join(
        save_path,
        "subprocess_gpu_{}".format(gpu)
    )
    objects_dict = config_handler.create_all({
        "save_path": save_path,
        "device": gpu,
        "world_size": world_size,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "is_distributed": True
    })
    manager = utils.get_default(objects_dict, "managers")
    manager.run()

def main():
    world_size = torch.cuda.device_count()
    batch_size = 120
    num_workers = 8
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, batch_size, num_workers))

if __name__ == "__main__":
    main()

# print("当前进程:{}".format(
#     os.getpid()
# ))