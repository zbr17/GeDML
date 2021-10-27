import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
from torchdistlog import logging
import pandas as pd 
import traceback

from ...core import models
from ..misc import utils

class BaseManager:
    """
    Manager all modules and computation devices. Support three kinds of computation:

    1. DataParallel (single machine)
    2. DistributedDataParallel (single machine)
    3. DistributedDataParallel (multi machines)
    """
    def __init__(
        self,
        trainer,
        tester,
        recorder,
        objects_dict,
        device=None,
        schedulers=None,
        gradclipper=None,
        samplers=None,
        is_resume=False,
        is_distributed=False,
        device_wrapper_type="DP",
        dist_url="tcp://localhost:23456",
        world_size=None,
        phase="train",
        primary_metric=["test", "recall_at_1"],
        to_device_list=["models", "collectors"],
        to_wrap_list=["models"],
        patience=10,
    ):
        self.trainer = trainer
        self.tester = tester
        self.recorder = recorder
        self.objects_dict = objects_dict
        self.device = device
        self.schedulers = schedulers
        self.gradclipper = gradclipper
        self.samplers = samplers
        self.epochs = -1
        self.is_resume = is_resume
        self.is_distributed = is_distributed
        self.device_wrapper_type = device_wrapper_type
        self.dist_url = dist_url
        self.world_size = world_size
        self.phase = phase
        self.primary_metric = primary_metric
        self.to_device_list = to_device_list
        self.to_wrap_list = to_wrap_list
        self.patience = patience

        self.best_metric = -1
        self.patience_counts = 0
        self.is_best = False

        self.assert_phase()
        self.assert_device()
        self.assert_required_member()
        self.assert_resume_folder_exist()

        self.initiate_objects_dict()
        self.initiate_members()
    
    @property
    def _required_member(self):
        return [
            "metrics",
            "collectors",
            "selectors",
            "models",
            "losses",
            "evaluators",
            "optimizers",
            "transforms",
            "datasets",
        ]
    
    def assert_phase(self):
        assert self.phase in ["train", "evaluate"]

    def assert_device(self):
        assert self.device_wrapper_type in ["DP", "DDP"]
        if self.is_distributed:
            assert self.device_wrapper_type == "DDP"

    def assert_required_member(self):
        object_dict_keys = list(self.objects_dict.keys())
        assert all(
            [item in object_dict_keys
            for item in self._required_member]
        )
    
    def assert_resume_folder_exist(self):
        if self.is_resume:
            assert not self.recorder.delete_old_folder

    def initiate_objects_dict(self):
        for k, v in self.objects_dict.items():
            setattr(self, k, v)
        del self.objects_dict
    
    def initiate_members(self):
        self.initiate_device()
        self.initiate_models()
        self.initiate_collectors()
        self.initiate_selectors()
        self.initiate_losses()
        self.initiate_schedulers()
        self.initiate_addition_items()
    
    def initiate_addition_items(self):
        pass

    def initiate_device(self):
        if isinstance(self.device, list):
            if len(self.device) > 1 and self.device_wrapper_type == "DDP" and not self.is_distributed:
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method=self.dist_url,
                    rank=0,
                    world_size=1
                )
        
        if self.is_distributed:
            self.world_size = (
                dist.get_world_size()
                if self.world_size is None
                else self.world_size
            )

        self.main_device_id, self.device_ids = None, None
        self.multi_gpu = False
        if self.device is None:
            self.main_device_id = 0
            self.device_ids = [0]
        elif isinstance(self.device, int):
            self.main_device_id = self.device
            self.device_ids = [self.device]
        elif isinstance(self.device, list):
            self.main_device_id = self.device[0]
            self.device_ids = self.device
            self.multi_gpu = (
                True if len(self.device_ids) > 1
                else False
            )
        else:
            raise TypeError(
                "Device type error!"
            )
        # initiate self.device
        self.device = torch.device(
            "cuda:{}".format(self.main_device_id)
            if torch.cuda.is_available()
            else "cpu"
        )
    
    def initiate_models(self):
        # to device
        is_to_device = "models" in self.to_device_list
        is_to_wrap = "models" in self.to_wrap_list
        if is_to_device:
            self._members_to_device("models", to_warp=is_to_wrap)
    
    def initiate_collectors(self):
        # to device
        is_to_device = "collectors" in self.to_device_list
        is_to_wrap = "collectors" in self.to_wrap_list
        if is_to_device:
            self._members_to_device("collectors", to_warp=is_to_wrap)
    
    def initiate_selectors(self):
        # to device
        is_to_device = "selectors" in self.to_device_list
        is_to_wrap = "selectors" in self.to_wrap_list
        if is_to_device:
            self._members_to_device("selectors", to_warp=is_to_wrap)

    def initiate_losses(self):
        # to device
        is_to_device = "losses" in self.to_device_list
        is_to_wrap = "losses" in self.to_wrap_list
        if is_to_device:
            self._members_to_device("losses", to_warp=is_to_wrap)
    
    def initiate_schedulers(self):
        if self.schedulers is None:
            self.schedulers = {}
    
    def _members_to_device(self, module_name: str, to_warp=True):
        members = getattr(self, module_name)
        # to device
        if not self.is_distributed:
            # single-device
            if self.multi_gpu:
                for k, v in members.items():
                    members[k] = members[k].to(self.device)
                    if to_warp:
                        if self.device_wrapper_type == "DP":
                            members[k] = torch.nn.DataParallel(
                                v,
                                device_ids=self.device_ids
                            )
                        else:
                            try:
                                members[k] = DDP(
                                    v,
                                    device_ids=self.device_ids,
                                    find_unused_parameters=True
                                )
                            except:
                                trace = traceback.format_exc()
                                logging.warning("{}".format(trace))
            else:
                for k, v in members.items():
                    members[k] = v.to(self.device)
        else:
            # multi-device
            for k, v in members.items():
                members[k] = members[k].to(self.device)
                try:
                    members[k] = DDP(
                        members[k], 
                        device_ids=self.device_ids,
                        find_unused_parameters=True
                    )
                except:
                    trace = traceback.format_exc()
                    logging.warning("{}".format(trace))
    
    """
    Run
    """
    
    def run(self, phase="train", start_epoch=0, total_epochs=61, is_test=True, is_save=True, interval=1, warm_up=2, warm_up_list=None):
        self.phase = phase
        self.assert_phase()
        self.prepare()
        self.maybe_resume(is_save=is_save)
        if self.phase == "train":
            for _ in range(start_epoch, total_epochs):
                # train phase
                self.epochs += 1
                if self.epochs < warm_up:
                    logging.info("Warm up with {}".format(warm_up_list))
                    self.trainer.set_activated_optims(warm_up_list)
                else:
                    self.trainer.set_activated_optims()
                self.trainer.train(epochs=self.epochs)
                self.release_memory()

                # test phase
                # to_test = True
                # if self.is_distributed:
                #     if dist.get_rank() != 0:
                #         to_test = False
                # if to_test:
                if is_test:
                    if (self.epochs % interval) == 0:
                        self.metrics = self.tester.test()
                        self.display_metrics()
                        self.save_metrics()
                        self.release_memory()
                if is_save:
                    self.save_models()
                
                # early stop
                if self.patience_counts >= self.patience:
                    logging.info("Training terminated!")
                    break
        elif self.phase == "evaluate":
            self.metrics = self.tester.test()
            self.display_metrics()
    
    def prepare(self):
        # prepare trainer
        utils.func_params_mediator(
            [self],
            self.trainer.prepare
        )
        # prepare tester
        utils.func_params_mediator(
            [
                {"recorders": self.recorder},
                self,
            ],
            self.tester.prepare
        )
        
    def maybe_resume(self, is_save=True):
        if self.is_resume:
            logging.info("Resume objects...")
            self.epochs = self.recorder.load_models(
                obj=self.trainer,
                device=self.device
            )
        else:
            if is_save:
                self.save_models(is_best=True)
    
    def meta_test(self):
        self.epochs = -1
        self.test()
        self.save_metrics()
        self.display_metrics()
    
    def save_metrics(self):
        for k, v in self.metrics.items():
            data, _ = self.recorder.get_data({k:v})
            self.recorder.update(data, self.epochs)
    
    def display_metrics(self):
        # best metric check
        cur_metric = self.metrics[self.primary_metric[0]][self.primary_metric[1]]
        if cur_metric > self.best_metric:
            self.best_metric = cur_metric
            self.is_best = True
            logging.info("NEW BEST METRIC!!!")
            self.patience_counts = 0
        else:
            self.is_best = False
            self.patience_counts += 1
        self.metrics[self.primary_metric[0]]["BEST_" + self.primary_metric[1]] = self.best_metric

        # display
        for k, v in self.metrics.items():
            logging.info("{} Metrics ---".format(k.upper()))
            if self.is_distributed:
                if dist.get_rank() == 0:
                    print(pd.DataFrame([v]))
            else:
                print(pd.DataFrame([v]))
        
    def save_models(self, is_best=None):
        if is_best is None:
            is_best = self.is_best
        self.recorder.save_models(self.trainer, step=self.epochs, best=is_best)
    
    def release_memory(self):
        torch.cuda.empty_cache()
