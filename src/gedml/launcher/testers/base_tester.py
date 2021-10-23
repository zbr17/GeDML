import torch
from torchdistlog import logging
from torchdistlog.tqdm import tqdm
import numpy as np 

from ..misc import utils

class BaseTester:
    """
    ``BaseTester`` takes charge of testing.

    Args:
        batch_size (int):
            Batch size.
        dataset_num_workers (int):
            Number of processes to load data.
        is_normalize (bool):
            Whether to normalize embeddings.
        splits_to_eval (list(str)):
            List of sub-dataset to evaluate.
        is_distributed (bool):
            Whether to set in the distributed mode.
    
    Example:
        >>> tester = BaseTester(32)
        >>> tester.prepare(...) # load models etc.
        >>> results = tester.test()
    """
    def __init__(
        self,
        batch_size,
        dataset_num_workers=8,
        is_normalize=True,
        splits_to_eval=["test"],
        is_distributed=False,
    ):
        self.batch_size = batch_size
        self.dataset_num_workers = dataset_num_workers
        self.is_normalize = is_normalize
        self.splits_to_eval = splits_to_eval
        self.is_distributed = is_distributed

        self.initiate_property()
    
    """
    Initialization
    """
    
    def initiate_property(self):
        self.trainable_object_list = [
            "models",
            "collectors",
        ]
        
    def initiate_datasets(self):
        self.datasets = {
            k: self.datasets[k] for k in self.splits_to_eval
        }
    
    """
    Set and Get
    """

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    """
    test
    """
    def prepare(
        self,
        models,
        datasets,
        evaluators,
        device,
        device_ids,
    ):
        """
        Load objects to be tested.

        Args:
            models (dict):
                Dictionary of models.
            datasets (dict):
                Dictionary of datasets.
            evaluators (dict):
                Dictioanry of evaluators.
            device (device):
                Computation device.
            device_ids (list(int)):
                Instruct Faiss package to use the corresponding devices.
        """
        # pass parameters
        self.models = models
        self.datasets = datasets
        self.evaluators = evaluators
        self.device = device
        self.device_ids = device_ids

    def test(self):
        """
        Start testing.

        Returns:
            dict: evaluation results.
        """
        self.initiate_datasets()
        # start to test
        self.set_to_eval()
        outputs = {}
        with torch.no_grad():
            for k, v in self.datasets.items():
                # get the dataset loader
                self.initiate_dataloader(dataset=v)
                # get the embeddings
                self.get_embeddings()
                # compute the metrics
                results = self.compute_metrics()
                outputs[k] = results
        return outputs
    
    def set_to_eval(self):
        for trainable_name in self.trainable_object_list:
            trainable_object = getattr(self, trainable_name, None)
            if trainable_object  is None:
                logging.warning(
                    "{} is not a member of trainer".format(
                        trainable_name
                    )
                )
            else:
                for v in trainable_object.values():
                    v.eval()

    def initiate_dataloader(self, dataset):
        logging.info(
            "{}: Initiating dataloader".format(
                self.__class__.__name__
            )
        )
        sampler = None
        # get dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(self.batch_size),
            sampler=sampler,
            drop_last=False,
            pin_memory=False,
            shuffle=False,
            num_workers=self.dataset_num_workers
        )
        self.dataloader_iter = iter(self.dataloader)
    
    def get_embeddings(self):
        logging.info(
            "Compute eval embeddings"
        )
        pbar = tqdm(self.dataloader_iter)
        embeddings_list, labels_list = [], []
        for info_dict in pbar:
            # get data
            data = info_dict["data"].to(self.device)
            label = info_dict["labels"].to(self.device)
            # forward
            embedding = self.compute_embeddings(data)
            embeddings_list.append(embedding)
            labels_list.append(label)
        self.embeddings = torch.cat(embeddings_list)
        self.labels = torch.cat(labels_list)

        # to numpy
        self.embeddings = self.embeddings.cpu().detach().numpy()
        self.labels = self.labels.cpu().numpy()
    
    def compute_embeddings(self, data):
        embedding = self.forward_models(data)
        return (
            torch.nn.functional.normalize(embedding, dim=-1) 
            if self.is_normalize 
            else embedding
        )
    
    def forward_models(self, data):
        embedding_trunk = self.models["trunk"](
            data
        )
        embedding_embedder = self.models["embedder"](
            embedding_trunk
        )
        return embedding_embedder
    
    def compute_metrics(self):
        metrics_dict = self.evaluators["default"].get_accuracy(
            self.embeddings,
            self.embeddings,
            self.labels,
            self.labels,
            True,
            device_ids=self.device_ids
        )
        return metrics_dict
    
