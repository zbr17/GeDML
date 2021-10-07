# from .metrics import *
from torchdistlog import logging
from . import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
import numpy as np 

class Calculator:
    """
    A evaluation manager class. Given ``query`` and ``referecne`` vectors, this module will return a evaluation dictionary including all evaluation results.

    Args:
        k_list (list):
            K-index list for ``recall at k`` and ``precision at k``.
        include (tuple):
            Evaluation functions must be included.
        exclude (tuple):
            Evaluation functions should be excluded.
        k (int):
            Parameter for knn search.

    Example:
        >>> calculator = Calculator(k_list=[1,2,4,8])
        >>> x = np.random.randn(100, 128)
        >>> labels = np.random.randint(0, 10, size=(100,))
        >>> output_dict = calculator.get_accuracy(x, x, labels, labels, True)
    """
    def __init__(self, k_list, include=(), exclude=(), k=None):
        assert isinstance(k_list, list)
        self.k_list = k_list
        self.k = k
        self.function_keyword = "calculate_"
        self.meta_function_keyword = "_meta_calculate_"
        self.include = include
        self.exclude = exclude

        self.init_original_function_dict()
        self.check_primary_metrics(include, exclude)
        self.original_function_dict = self.get_function_dict(include, exclude)
        self.curr_function_dict = self.get_function_dict()
    
    def check_primary_metrics(self, include=(), exclude=()):
        primary_metrics = list(self.original_function_dict.keys())
        for met in [include, exclude]:
            if not isinstance(met, (tuple, list)):
                raise TypeError("Arguments must be of type tuple, not {}.".format(type(met)))
            if not set(met).issubset(set(primary_metrics)):
                raise ValueError("Primary metrics must be one or more of: {}.".format(primary_metrics))

    def meta_metrics(self):
        return ['precision_at_k', 'recall_at_k']
    
    def requires_clustering(self):
        return ["NMI", "AMI", "f1_score"]

    def requires_knn(self): 
        return ["mean_average_precision_at_r", "r_precision"]
    
    def init_original_function_dict(self):
        # prepare other metrics except recall@k and precision@k
        function_names = [x for x in dir(self) if x.startswith(self.function_keyword)]
        metrics = [x.replace(self.function_keyword, "", 1) for x in function_names]
        self.original_function_dict = {x:getattr(self, y) for x, y in zip(metrics, function_names)}

        # get recall and precision function
        for meta_metric in self.meta_metrics():
            meta_metric_names, meta_metric_functions = [], []
            for k in self.k_list:
                meta_metric_names.append(meta_metric.replace('at_k', "at_"+str(k)))
                meta_metric_functions.append(getattr(self, self.meta_function_keyword+meta_metric)(k))
            self.original_function_dict.update(zip(meta_metric_names, meta_metric_functions))
    
    def get_function_dict(self, include=(), exclude=()):
        if len(include) == 0:
            include = list(self.original_function_dict.keys())
        include_metrics = [k for k in include if k not in exclude]
        return {k:v for k, v in self.original_function_dict.items() if k in include_metrics}
    
    def get_curr_metrics(self):
        return [k for k in self.curr_function_dict.keys()]
    
    def get_cluster_labels(self, query, query_labels, device_ids=None, **kwargs):
        num_clusters = len(np.unique(query_labels.flatten()))
        return metrics.run_kmeans(query, num_clusters, device_ids=device_ids)
    
    def _meta_calculate_precision_at_k(self, k):
        def calculate_precision_at_k(knn_labels, query_labels, not_lone_query_mask, **kwargs):
            if not any(not_lone_query_mask):
                return 0
            knn_labels, query_labels = knn_labels[not_lone_query_mask], query_labels[not_lone_query_mask]
            return metrics.precision_at_k(knn_labels, query_labels[:, None], k)
        return calculate_precision_at_k
    
    def _meta_calculate_recall_at_k(self, k):
        def calculate_recall_at_k(knn_labels, query_labels, not_lone_query_mask, **kwargs):
            if not any(not_lone_query_mask):
                return 0
            knn_labels, query_labels = knn_labels[not_lone_query_mask], query_labels[not_lone_query_mask]
            return metrics.recall_at_k(knn_labels, query_labels, k)
        return calculate_recall_at_k
    
    def calculate_NMI(self, query_labels, cluster_labels, **kwargs):
        return normalized_mutual_info_score(query_labels, cluster_labels)
    
    def calculate_AMI(self, query_labels, cluster_labels, **kwargs):
        return adjusted_mutual_info_score(query_labels, cluster_labels)
    
    def calculate_f1_score(self, query_labels, cluster_labels, **kwargs):
        return metrics.f1_score(query_labels, cluster_labels)
    
    def calculate_mean_average_precision_at_r(self, knn_labels, query_labels, not_lone_query_mask, embeds_same_source, label_counts, **kwargs):
        if not any(not_lone_query_mask):
            return 0
        knn_labels, query_labels = knn_labels[not_lone_query_mask], query_labels[not_lone_query_mask]
        return metrics.mean_average_precision_at_r(knn_labels, query_labels[:, None], embeds_same_source, label_counts)
    
    def calculate_r_precision(self, knn_labels, query_labels, not_lone_query_mask, embeds_same_source, label_counts, **kwargs):
        if not any(not_lone_query_mask):
            return 0
        knn_labels, query_labels = knn_labels[not_lone_query_mask], query_labels[not_lone_query_mask]
        return metrics.r_precision(knn_labels, query_labels[:, None], embeds_same_source, label_counts)
    
    def get_accuracy(self, query: np.ndarray, reference: np.ndarray, query_labels: np.ndarray, reference_labels: np.ndarray, embeds_same_source: bool, include=(), exclude=(), device_ids=None):
        """
        Compute all evaluation indicators. 

        Args:
            query (np.ndarray):
                Samples to be tested (or without labels). size: :math:`B_1 \\times dim`.
            reference (np.ndarray):
                Samples working as reference. size: :math:`B_2 \\times dim`.
            query_labels (np.ndarray):
                Queris' labels. size: :math:`B_1`.
            reference_labels (np.ndarray):
                References' labels. size: :math:`B_2`.
            embeds_same_source (bool):
                Whether ``query`` and ``reference`` data are from the same source.
            include (tuple):
                Evaluation functions to include.
            exclude (tuple):
                Evaluation functions to exclude.
            device_ids (list): 
                Device indices to call by Faiss package. default: None.
            
        Returns:
            dict: A dictionary which contains all results.
        """
        logging.info('Start computing metrics!...')
        embeds_same_source = embeds_same_source or (query is reference)
        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {
            "query": query,
            "reference": reference,
            "query_labels": query_labels,
            "reference_labels": reference_labels,
            "embeds_same_source": embeds_same_source
        }

        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            label_counts, num_k = metrics.get_label_counts(reference_labels)
            if self.k is not None: num_k = self.k
            assert num_k >= max(self.k_list) # max(k) mustn't samller than the max element of k_list
            knn_indices, knn_distances = metrics.get_knn(reference, query, num_k, embeds_same_source, device_ids=device_ids)
            knn_labels = reference_labels[knn_indices]
            lone_query_labels = metrics.get_lone_query_labels(query_labels, reference_labels, label_counts, embeds_same_source)
            not_lone_query_mask = ~np.isin(query_labels, lone_query_labels)

            if not any(not_lone_query_mask):
                logging.warning("None of the query labels are in the reference set.")
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels
            kwargs["knn_distances"] = knn_distances
            kwargs["lone_query_labels"] = lone_query_labels
            kwargs["not_lone_query_mask"] = not_lone_query_mask
            kwargs["device_ids"] = device_ids
        
        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs) 
        
        return self._get_accuracy(self.curr_function_dict, **kwargs)
    
    def _get_accuracy(self, function_dict, **kwargs):
        return {k:v(**kwargs) for k,v in function_dict.items()}
                
