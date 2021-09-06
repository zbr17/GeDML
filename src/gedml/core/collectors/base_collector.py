import torch
from abc import ABCMeta, abstractmethod

from ..modules import WithRecorder

class BaseCollector(WithRecorder, metaclass=ABCMeta):
    """
    Base class of collector module, which defines main collector method in function ``collect`` and ``update``, and defines default parameters in function ``output_list``, ``input_list`` and ``_default_next_module``.

    Args:
        metric (metric instance):
            metric to compute matrix (e.g. euclidean or cosine)
    
    Example:
        >>> metric = MetricFactory(is_normalize=True, metric_name="cosine")
        >>> data = torch.randn(10, 3, 227, 227)
        >>> embeddings = torch.randn(10, 128)
        >>> labels = torch.randint(0, 3, size=(10,))
        >>> collector = DefaultCollector(metric=metric)
        >>> # collector forward
        >>> output_dict = collector(data, embeddings, labels)
    """
    def __init__(self, metric, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
    
    @property
    def is_global_collector(self):
        return False
    
    def update(self, *args, **kwargs):
        """
        Define the interface that collector can update itself by giving specific information (default do nothing)
        """
        pass

    def forward(
        self, 
        data, 
        embeddings, 
        labels
    ) -> tuple:
        """
        In ``collect`` function, three kinds of operation may be done:

        1. maintain sets of parameters about collecting (or synthesizing) samples
        2. compute metric matrix and pass to next module
        3. compute some regularization term using embeddings

        Args:
            data (torch.Tensor):
                Images with RGB channels. size: :math:`B \\times C \\times H \\times W`
            embeddings (torch.Tensor):
                Embedding. size: :math:`B \\times dim`
            lables (torch.Tensor): 
                Ground truth of dataset. size: :math:`B \\times 1`

        Returns:
            tuple: include metric matrix, labels etc according to function ``output_list``.

            Let :math:`B_{row}` be the length of rows and :math:`B_{col}` be the length of columns, typical output type is listed below:

            1. metric matrix (torch.Tensor): size: :math:`B_{row} \\times B_{col}`
            2. labels of rows (torch.Tensor): size: :math:`B_{row} \\times 1` or :math:`B_{row} \\times B_{col}`
            3. labels of columns (torch.Tensor): size: :math:`1 \\times B_{col}` or :math:`B_{row} \\times B_{col}`
            4. is_from_same_source (bool): indicate whether row vectors and column vectors are from the same data
        
        """
        metric_mat, labels, proxies_labels, is_same_source = None, None, None, False
        return (
            metric_mat, 
            labels, 
            proxies_labels, 
            is_same_source
        )
