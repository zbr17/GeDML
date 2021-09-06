import torch
import torch.distributed as dist
import logging
from tqdm import tqdm

from ..misc import utils, Storage
from .loss_handler import LossHandler
from ...config.setting.recorder_setting import (
    TO_RECORD_LIST,
    TO_SAVE_LIST
)

class BaseTrainer:
    """
    ``BaseTrainer`` takes charge of training.

    ``storage`` holds all the intermediate variables such as ``data``, ``embeddings``, etc. All other modules, such as ``collectors``, ``selectors``, etc, access thier own parameter list to get the corresponding parameters.

    ``loss_handler`` will compute weighted loss.

    Args:
        batch_size (int):
            Batch size.
        freeze_trunk_batchnorm (bool):
            Whether to freeze batch normalization layer.
        dataset_num_workers (int):
            Number of process to load data.
        is_distributed (bool):
            Whether to use distibuted mode.
    """
    def __init__(
        self,
        batch_size,
        wrapper_params,
        freeze_trunk_batchnorm=False,
        dataset_num_workers=8,
        is_distributed=False,
    ):
        self.batch_size = batch_size
        self.freeze_trunk_batchnorm = freeze_trunk_batchnorm
        self.dataset_num_workers = dataset_num_workers
        self.is_distributed = is_distributed
        
        self.epochs = 0
        self.storage = Storage(wrapper_params)
        self.initiate_property()
    
    """
    Initialization
    """
    
    def initiate_property(self):
        # to save list (for recorders)
        setattr(
            self,
            TO_SAVE_LIST,
            ["models", "collectors", "losses"]
        )
        # recordable list
        self.recordable_object_list = ["models"]
        # trainable list
        self.trainable_object_list = [
            "models",
            "collectors"
        ]
        
    @property
    def models_forward_order(self):
        return ["trunk", "embedder"]
    
    """
    Set and Get
    """

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_distributed(self, flag=True):
        self.is_distributed = flag
    
    def set_activated_optims(self, optim_list=None):
        self.optim_list = (
            list(self.optimizers.keys()) 
            if optim_list is None
            else optim_list
        )
        logging.info("Set activated optims: {}".format(self.optim_list))
        
    """
    Train
    """

    def prepare(
        self,
        collectors,
        selectors,
        losses,
        models,
        optimizers,
        datasets,
        schedulers=None,
        gradclipper=None,
        samplers=None,
        collatefns=None,
        device=None,
        recorders=None
    ):
        """
        Load modules to prepare training.
        """
        # pass parameters
        self.collectors = collectors
        self.selectors = selectors
        self.losses = losses
        self.models = models
        self.optimizers = optimizers
        self.datasets = datasets
        self.schedulers = schedulers
        self.gradclipper = gradclipper
        self.samplers = samplers
        self.collatefns = collatefns
        self.device = device
        self.loss_handler = LossHandler()
        self.recorders = utils.get_default(recorders, "recorders")

        self.optim_list = list(self.optimizers.keys())


    def train(self, epochs=None):
        """
        Start training.

        Args:
            epochs (int):
                Epoch to start.
        """
        # start to train
        if epochs is not None:
            self.epochs = epochs
        self.initiate_dataloader()
        logging.info("TRAIN EPOCH: {}".format(self.epochs))
        self.pipeline()
        self.epochs += 1
        # self.release_memory()
    
    def pipeline(self):
        # use global collector 
        self.use_global_collector()
        self.set_to_train()
        self.show_lr()

        # start training
        self.pbar = tqdm(range(self.iterations_per_epoch))
        for self.iteration in self.pbar:
            self.prepare_forward()
            self.forward_pipeline()
            self.backward_and_update()
            self.update_record(self.recorders)
            self.pbar.set_description(
                "Loss={:.4f}".format(
                    self.loss_handler.get_total(),
                )
            )
        self.loss_handler.average_losses()
        self.step_schedulers(metrics=self.loss_handler.get_total())
    
    def show_lr(self):
        default_idx = 0
        for k, v in self.optimizers.items():
            lr = v.param_groups[default_idx]["lr"]
            logging.info("{} optimizer's lr: {}".format(k, lr))
    
    def forward_pipeline(self):
        self.forward_models()
        self.forward_collectors()
        self.forward_selectors()
        self.forward_losses()
    
    def initiate_dataloader(self):
        logging.info(
            "{}: Initiating dataloader".format(
                self.__class__.__name__
            )
        )
        
        # more initialization
        self.sampler, self.collate_fn = None, None
        self._initiate_dataloader_sampler_collatefn()

        # get dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.datasets["train"],
            batch_size=int(self.batch_size),
            sampler=self.sampler,
            drop_last=True,
            pin_memory=False,
            shuffle=self.samplers is None,
            num_workers=self.dataset_num_workers,
            collate_fn=self.collate_fn
        )
        self.iterations_per_epoch = len(self.dataloader)
        self.dataloader_iter = iter(self.dataloader)
    
    def _initiate_dataloader_sampler_collatefn(self):
        # extract the sampler
        if self.is_distributed:
            if self.samplers is not None:
                logging.warn("class samplers can't be used in distribued training mode!")
            self.samplers = {}
            self.samplers["train"] = torch.utils.data.distributed.DistributedSampler(
                dataset=self.datasets["train"],
                shuffle=True,
            )
            self.sampler = self.samplers["train"]
            logging.info("Get distributed sampler {}".format(
                self.sampler.__class__.__name__,
            ))
        else:
            self.sampler = (
                self.samplers["train"] 
                if self.samplers is not None
                else None
            )
        
        # extract the collate_fn
        self.collate_fn = (
            self.collatefns["train"]
            if self.collatefns is not None
            else None
        )

    def set_to_train(self):
        for trainable_name in self.trainable_object_list:
            trainable_object = getattr(self, trainable_name, None)
            if trainable_object is None:
                logging.warn(
                    "{} is not a member of trainer".format(
                        trainable_name
                    )
                )
            else:
                for v in trainable_object.values():
                    v.train()
        # maybe some parameters should be frozen
        self._maybe_freeze_batchnorm()
    
    def _maybe_freeze_batchnorm(self):
        if self.freeze_trunk_batchnorm:
            self.models["trunk"].apply(
                utils.set_layers_to_eval("BatchNorm")
            )
        
    def use_global_collector(self):
        for collector in self.collectors.values():
            if collector.is_global_collector:
                logging.info("Global collector updating...")
                collector.global_update(self)
                self.release_memory()
    
    def prepare_forward(self):
        # set sampler
        self._prepare_forward_set_sampler()

        # zero loss-values
        self.loss_handler.zero_losses()

        # zero grad
        for v in self.models.values():
            v.zero_grad()
        for v in self.optimizers.values():
            v.zero_grad()

        # get batch
        self._prepare_forward_get_batch()
    
    def _prepare_forward_set_sampler(self):
        if self.is_distributed:
            self.sampler.set_epoch(self.epochs)
    
    @property
    def info_dict_to_device(self):
        return [
            "data", 
            "labels"
        ]
    
    def _prepare_forward_get_batch(self):
        info_dict = next(self.dataloader_iter)
        for key in info_dict.keys():
            setattr(self.storage, key, info_dict[key])
        self.storage.tensors_to_device(self.info_dict_to_device, self.device)

    def forward_models(self):
        # get data and labels
        data = self.storage.get("data")
        labels = self.storage.get("labels")

        # forward backbone (trunk model)
        features = self.models["trunk"](data)

        # if distributed
        if self.is_distributed:
            features, labels = utils.distributed_gather_objects(
                features, labels
            )

        self.storage.features = features
        self.storage.labels = labels
        self.storage.indices_dict["models"] = {"embedder": {"": {}}}

        self.storage.update(self.models["embedder"], cur_module="models")

    def forward_collectors(self):
        # update collector
        self.update_collectors()
        # forward collector
        self.storage.update(self.collectors, cur_module="collectors")
    
    def update_collectors(self):
        for collector in self.collectors.values():
            if isinstance(collector, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                collector.module.update(self)
            else:
                collector.update(self)
    
    def forward_selectors(self):
        self.storage.update(self.selectors, cur_module="selectors")

    def forward_losses(self):
        value_dict = self.storage.update(self.losses, cur_module="losses")

        # update loss_values
        self.loss_handler.update_losses(self.storage.return_loss_dict())

    def backward_and_update(self):
        self._backward_and_update_preprocess()

        self.loss_handler.backward()
        # clip gradients
        if self.gradclipper is not None:
            for v in self.gradclipper.values():
                v()
        # step optimizers
        for k in self.optim_list:
            self.optimizers[k].step()
    
    def _backward_and_update_preprocess(self):
        # for distributed
        if self.is_distributed:
            # dist.barrier()
            dist.all_reduce(
                tensor=self.loss_values["total_loss"],
                op=dist.ReduceOp.SUM
            )

    def step_schedulers(self, **kwargs):
        """
        All schedulers step at the end of each epoch for the moment
        """
        if self.schedulers is not None:
            for k in self.optim_list:
                v = self.schedulers[k]
                if isinstance(v, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    v.step(**kwargs)
                else:
                    v.step()

    def update_record(self, recorders=None):
        if recorders is not None:
            total_iterations = (
                self.epochs * self.iterations_per_epoch 
                + self.iteration
            )
            # update loss
            self.loss_handler.record_losses(recorders, total_iterations)

            # update other statistics
            for recordable_name in self.recordable_object_list:
                recordable_object = getattr(self, recordable_name)
                for k, v in recordable_object.items():
                    if k == "trunk": # TODO:
                        continue
                    to_record_obj = (
                        v.module
                        if isinstance(v, torch.nn.DataParallel) or isinstance(v, torch.nn.parallel.DistributedDataParallel)
                        else v
                    )
                    data, _ = recorders.get_data(to_record_obj, k)
                    recorders.update(data, total_iterations)

    def release_memory(self):
        torch.cuda.empty_cache()