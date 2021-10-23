import torch
from torchdistlog import logging
from torchdistlog.tqdm import tqdm
import torch.nn.functional as F 

from .. import ProxyCollector
from ._default_global_collector import _DefaultGlobalCollector

class GlobalProxyCollector(ProxyCollector, _DefaultGlobalCollector):
    """
    Compute the global proxies before updating other parameters.
    """

    def __init__(
        self,
        optimizer_name="Adam",
        optimizer_param={"lr": 0.001},
        dataloader_param={"batch_size": 120, "drop_last": False, "shuffle": True, "num_workers": 8},
        max_iter=50000,
        error_bound=1e-3,
        total_patience=10,
        auth_weight=1.0,
        repre_weight=1.0,
        disc_weight=1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_name = optimizer_name
        self.optimizer_param = optimizer_param
        self.dataloader_param = dataloader_param
        self.max_iter = max_iter
        self.error_bound = error_bound
        self.total_patience = total_patience
        self.auth_weight = auth_weight
        self.repre_weight = repre_weight
        self.disc_weight = disc_weight

        self.datasets, self.models = None, None

    def global_update(self, trainer):
        logging.info("Start to compute global embeddings")
        self.prepare(trainer)
        self.compute_global_embedding()
        self.compute_global_proxy()
    
    def prepare(self, trainer):
        _DefaultGlobalCollector.prepare(self, trainer)

        # get optimizer
        self.optimizer_param["params"] = [self.proxies]
        self.optimizer = getattr(
            torch.optim, 
            self.optimizer_name,
        )(**self.optimizer_param)
    
    def compute_global_proxy(self):
        pre_loss_value = -100
        patience_count = 0
        pbar = tqdm()
        index = 0
        while(True):
            # update proxies
            index += 1
            self.optimizer.zero_grad()
            cur_loss = self.loss_func(self.embeddings, self.labels)
            cur_loss.backward()
            self.optimizer.step()

            # raise error when get max_iter
            if index > self.max_iter:
                logging.warning("Reach MAX_ITER!")
                raise RuntimeError()

            # termination condition
            cur_error = abs(cur_loss.item() - pre_loss_value)
            pbar.set_description("CurrError={}, index={}".format(
                cur_error, index
            ))
            pre_loss_value = cur_loss.item()
            if cur_error < self.error_bound:
                patience_count += 1
                if patience_count > self.total_patience:
                    pbar.close()
                    logging.info("Have reached the solution of proxies!")
                    break
            else:
                patience_count = 0

    def loss_func(self, embeddings, labels):
        dtype = embeddings.dtype
        # normalize
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        proxies = F.normalize(self.proxies, dim=-1, p=2)

        # compute cosine matrix
        metric_mat = torch.matmul(embeddings, proxies.t())
        pos_mask = (labels.unsqueeze(1) == self.proxy_labels.unsqueeze(0)).byte()

        # compute loss-auth
        masked_metric_mat = metric_mat * pos_mask
        masked_metric_mat[pos_mask == 0] = torch.finfo(dtype).min
        loss_auth = torch.mean(torch.max(masked_metric_mat, dim=0)[0])

        # compute loss-repre
        loss_repre = torch.mean(torch.max(masked_metric_mat, dim=1)[0])

        if self.centers_per_class > 1 and self.disc_weight > 0:
            # compute loss-disc
            proxy_metric_mat = torch.matmul(proxies, proxies.t())
            proxy_pos_mask = (self.proxy_labels.unsqueeze(1) == self.proxy_labels.unsqueeze(0)).byte()
            proxy_neg_mask = proxy_pos_mask ^ 1
            proxy_pos_mask.fill_diagonal_(0)
            
            ## compute pos metric
            pos_sum_exp = torch.sum(
                torch.exp(
                    - proxy_metric_mat
                ) * proxy_pos_mask, dim=-1
            )
            neg_sum_exp = torch.sum(
                torch.exp(
                    proxy_metric_mat
                ) * proxy_neg_mask, dim=-1
            )
            loss_disc = torch.mean(
                torch.log(
                    1 + pos_sum_exp * neg_sum_exp
                )
            )
        else:
            loss_disc = 0
        
        return (
            - self.auth_weight * loss_auth -
            self.repre_weight * loss_repre + 
            self.disc_weight * loss_disc
        )

    