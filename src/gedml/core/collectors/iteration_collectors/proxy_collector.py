import torch
import math
from ..base_collector import BaseCollector

class ProxyCollector(BaseCollector):
    """
    Maintain proxy parameters to support proxy-based metric learning methods.

    Args:
        num_classes (int): Number of classes. default: 100.
        embeddings_dim (int): Dimension of embeddings. default: 128.
        centers_per_class (int): Number of centers per class. default: 1
    """
    def __init__(
        self,
        num_classes=100,
        embeddings_dim=128,
        centers_per_class=1,
        regularize_func="softtriple",
        regularize_weight=0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.embeddings_dim = embeddings_dim
        self.centers_per_class = centers_per_class
        self.regularize_func = regularize_func
        self.regularize_weight = regularize_weight

        self.initiate_regularize()
        self.initiate_params()

    def initiate_regularize(self):
        if isinstance(self.regularize_func, str):
            if self.regularize_func == "softtriple":
                self.register_buffer(
                    "pos_mask",
                    torch.zeros(self.num_classes*self.centers_per_class,
                    self.num_classes*self.centers_per_class, dtype=torch.bool)
                )
                K = self.centers_per_class
                for i in range(self.num_classes):
                    for j in range(K):
                        self.pos_mask[
                            i*K+j,
                            (i*K+j+1):(i*K+K)
                        ] = 1
                self.regularize_func = self._regularize_softtriple
            elif self.regularize_func == "structural":
                self.regularize_func = self._regularize_structural
    
    def _regularize_softtriple(self):
        proxies = torch.nn.functional.normalize(self.proxies, dim=-1, p=2)
        sim_mat_proxy = torch.matmul(proxies, proxies.t())
        reg_loss = torch.sum(
            torch.sqrt(2.0 + 1e-5 - 2. * sim_mat_proxy[self.pos_mask])
        ) / (self.num_classes * self.centers_per_class * (self.centers_per_class - 1))
        return reg_loss
    
    def _regularize_structural(self):
        proxies = torch.nn.functional.normalize(self.proxies, dim=-1, p=2)
        proxy_metric_mat = torch.matmul(proxies, proxies.t())
        proxy_pos_mask = (self.proxy_labels.unsqueeze(1) == self.proxy_labels.unsqueeze(0)).byte()
        proxy_neg_mask = proxy_pos_mask ^ 1
        proxy_pos_mask.fill_diagonal_(0)

        # compute pos metric
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
        reg_loss = torch.mean(
            torch.log(
                1 + pos_sum_exp * neg_sum_exp
            )
        )
        return reg_loss

    
    def initiate_params(self):
        """
        Initiate proxies.
        """
        self.proxies = torch.nn.Parameter(
            torch.randn(
                self.num_classes * self.centers_per_class,
                self.embeddings_dim
            )
        )
        
        proxy_labels = (
            torch.arange(self.num_classes)
            .unsqueeze(1)
            .repeat(1, self.centers_per_class)
        ).flatten()
        self.register_buffer("proxy_labels", proxy_labels)
        torch.nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))
    
    def forward(self, data, embeddings, labels) -> tuple:
        """
        Compute similarity (or distance) matrix between embeddings and proxies.
        """
        metric_mat = self.metric(embeddings, self.proxies)
        is_same_source = False

        # regularize multi-proxy
        if self.regularize_weight > 0 and self.centers_per_class > 1:
            reg_loss = self.regularize_func()
        else:
            reg_loss = 0
        return (
            metric_mat,
            labels.unsqueeze(-1),
            self.proxy_labels.unsqueeze(0),
            is_same_source,
            reg_loss,
            self.regularize_weight
        )
