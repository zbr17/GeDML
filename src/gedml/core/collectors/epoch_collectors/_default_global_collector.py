import torch
from torchdistlog.tqdm import tqdm

class _DefaultGlobalCollector:
    def __init__(
        self,
        dataloader_param={
            "batch_size": 120, 
            "drop_last": False,
            "shuffle": True,
            "num_workers": 8
        },
        *args,
        **kwargs
    ):
        self.dataloader_param = dataloader_param
    
    @property
    def is_global_collector(self):
        return True
    
    def global_update(self, trainer):
        raise NotImplementedError()

    def prepare(self, trainer):
        self.datasets = trainer.datasets
        self.models = trainer.models
        self.device = trainer.device

        # get dataloader
        self.dataloader_param["dataset"] = self.datasets["train"]
        self.data_loader = torch.utils.data.DataLoader(**self.dataloader_param)
        self.data_loader = iter(self.data_loader)

        # set model
        self.set_to_eval()
    
    def set_to_eval(self):
        for v in self.models.values():
            v.eval()
    
    def compute_global_embedding(self):
        embeddings_list, labels_list = [], []
        with torch.no_grad():
            pbar = tqdm(self.data_loader)
            for info_dict in pbar:
                data, label = info_dict["data"].to(self.device), info_dict["labels"].to(self.device)
                features = self.models["trunk"](data)
                embeddings = self.models["embedder"](features)
                embeddings_list.append(embeddings)
                labels_list.append(label)
        self.embeddings = torch.cat(embeddings_list, dim=0).to(self.device).detach()
        self.labels = torch.cat(labels_list, dim=0).to(self.device)