import torch
import torch.nn.functional as F 
from . import (
    distance_metrics,
    similarity_metrics
)

class MetricFactory(torch.nn.Module):
    """
    Get different metric (distance or similarity)

    Args:
        is_normalize (bool):
            Whether to normalize the embeddings
        metric_name (str):
            'euclid', 'cosine', etc.
    
    Example:
        Get ``euclid`` metric:

        >>> metric = MetricFactory(is_normalize=False, metric_name="euclid")
        >>> data = torch.randn(100, 128)
        >>> matrix = metric(data, data)
        
    """
    def __init__(self, is_normalize, metric_name, addition=None, **kwargs):
        super().__init__(**kwargs)
        self.is_normalize = is_normalize
        self.metric_name = metric_name
        
        self.init_metric(addition)
    
    def init_metric(self, addition):
        self.metric_func = None
        for key in globals().keys():
            if "_metrics" in key:
                metric_module = globals()[key]
                self.metric_func = getattr(metric_module, self.metric_name, None)
                if self.metric_func is not None:
                    break
        assert self.metric_func is not None, "{} isn't valid! Please check 'metric_name'!"
        if addition is None:
            self.metric_func = self.metric_func()
        else:
            self.metric_func = self.metric_func(**addition)
        self.metric_type = self.metric_func.metric_type
        
    def forward(self, *args) -> torch.Tensor:
        """
        Get metric matrix.

        Args:
            *args (sequence):
                Sequence which is used to compute matrix.
        
        Returns:
            torch.Tensor: metric matrix.
        """
        args = list(args)
        if self.is_normalize:
            for idx, _ in enumerate(args):
                args[idx] = F.normalize(args[idx], dim=-1)
        metric_output = self.metric_func(*tuple(args))
        return metric_output
    

    