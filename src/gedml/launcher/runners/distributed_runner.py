import logging
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "10002"

import torch
import torch.distributed as dist 
import torch.backends.cudnn as cudnn
torch.multiprocessing.set_sharing_strategy('file_system')

from ..creators import ConfigHandler
from ..misc import utils

# TODO: to be perfected

class DistributedRunner:
    def __init__(
        self,
        opt
    ):
        self.opt = opt
        self.assert_opt()
    
    @property
    def required_members_list(self):
        return ["gpu", "world_size", "rank", "batch_size", "num_workers"]
    
    def assert_opt(self):
        # assert
        assert all(
            [
                getattr(self.opt, item, None) is not None
                for item in self.required_members_list
            ]
        )

    def run(self):
        cudnn.deterministic = True
        cudnn.benchmark = True
        # initiate torch.distributed
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.opt.world_size,
            rank=self.opt.rank
        )

        # get config_handler
        config_handler = ConfigHandler(
            link_path=self.opt.link_path
        )
        del self.opt.link_path

        # change link config
        if self.opt.dataset == "CUB200":
            config_handler.link_config["datasets"] = [
                {"train": "cub200_train.yaml"},
                {"test": "cub200_test.yaml"}
            ]
        elif self.opt.dataset == "ImageNet":
            config_handler.link_config["datasets"] = [
                {"train": "imagenet_train.yaml"},
                {"test": "imagenet_test.yaml"}
            ]
        del self.opt.dataset
        
        # initiate params_dict
        config_handler.get_params_dict()

        # modify parameters: save_path, gpu_device
        save_path = config_handler.get_certain_params_dict(["save_path"])
        save_path = utils.get_first_dict_value(save_path["save_path"])
        save_path = os.path.join(
            save_path,
            "subprocess_rank_{}".format(self.opt.rank)
        )
        del self.opt.rank

        modify_dict = {
            "device": self.opt.gpu,
            "is_distributed": True
        }
        del self.opt.gpu
        self.opt = self.opt.__dict__
        modify_dict.update(self.opt)
        objects_dict = config_handler.create_all(modify_dict)
        manager = utils.get_default(objects_dict, "managers")
        manager.run()
