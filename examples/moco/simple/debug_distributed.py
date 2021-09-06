import sys
import logging
import yaml
import os
from os.path import join as opj
workspace = os.environ["WORKSPACE"]
sys.path.append(
    opj(workspace, 'code/GeDML/src')
)
logging.getLogger().setLevel(logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "10002"

import torch
import torch.multiprocessing as mp 
import torch.distributed as dist 
import torchvision.transforms as transforms
torch.backends.cudnn.deterministic = True

import gedml
from gedml.launcher.creators import ConfigHandler
from gedml.launcher.misc import utils

def main_worker(gpu, world_size, batch_size, num_workers):
    # initiate torch.distributed
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=gpu
    )

    # get config_handler
    config_handler = ConfigHandler(
        link_path=os.path.join(workspace, "code/Experiments/GeDML/demo/moco/link.yaml")
    )

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
    batch_size = 64
    num_workers = 0
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, batch_size, num_workers))

if __name__ == "__main__":
    main()
