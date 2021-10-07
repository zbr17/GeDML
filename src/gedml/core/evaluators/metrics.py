import numpy as np 
import torch
from torchdistlog import logging
from scipy.special import comb
# sklearn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
# faiss
try:
    import faiss
except ModuleNotFoundError:
    logging.warning("Faiss Package Not Found! (metrics package)")

from ..misc import utils

# cluster algorithm
def get_knn(ref_embeds, embeds, k, embeds_same_source=False, device_ids=None):
    d = ref_embeds.shape[1]
    if device_ids is not None:
        index = faiss.IndexFlatL2(d)
        index = utils.index_cpu_to_gpu_multiple(index, gpu_ids=device_ids)
        index.add(ref_embeds)
        distances, indices = index.search(embeds, k+1)
        if embeds_same_source:
            return indices[:, 1:], distances[:, 1:]
        else:
            return indices[:, :k], distances[:, :k]
    else:
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(ref_embeds)
        distances, indices = neigh.kneighbors(embeds, k + 1)
        if embeds_same_source:
            return indices[:, 1:], distances[:, 1:]
        else:
            return indices[:, :k], distances[:, :k]

def get_knn_from_mat(metric_mat, k, embeds_same_source=False, is_min=True, device_ids=None):
    device = torch.device("cuda:{}".format(device_ids[0]))
    metric_mat = torch.from_numpy(metric_mat).to(device)
    # sort
    sorted_value, sorted_indices = torch.sort(metric_mat, dim=-1, descending=not is_min)
    if embeds_same_source:
        return (
            sorted_indices[:, 1:(k+1)].cpu().numpy(),
            sorted_value[:, 1:(k+1)].cpu().numpy()
        )
    else:
        return (
            sorted_indices[:, :k].cpu().numpy(),
            sorted_value[:, :k].cpu().numpy()
        )

def run_kmeans(x, num_clusters, device_ids=None):
    _, d = x.shape
    if device_ids is not None:
        # faiss implementation of k-means
        clus = faiss.Clustering(d, num_clusters)
        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        index = faiss.IndexFlatL2(d)
        index = utils.index_cpu_to_gpu_multiple(index, gpu_ids=device_ids)
        # perform the training
        clus.train(x, index)
        _, idxs = index.search(x, 1)
        return np.array([int(n[0]) for n in idxs], dtype=np.int64)
    else:
        # k-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
        return kmeans.labels_ 

def run_pca(x, out_dim, device_ids=None):
    if device_ids is not None:
        mat = faiss.PCAMatrix(x.shape[1], out_dim)
        mat.train(x)
        assert mat.is_trained
        return mat.apply_py(x)
    else:
        pca = PCA(n_components=out_dim)
        data_output = pca.fit_transform(x)
        return data_output

# metrics functions: code from: github: pytorch-metric-learning
def get_relevance_mask(shape, gt_labels, embeds_same_source, label_counts):
    relevance_mask = np.zeros(shape=shape, dtype=np.int)
    for k, v in label_counts.items():
        matching_rows = np.where(gt_labels==k)[0]
        max_column = v-1 if embeds_same_source else v
        relevance_mask[matching_rows, :max_column] = 1
    return relevance_mask

def get_label_counts(ref_labels):
    unique_labels, label_counts = np.unique(ref_labels, return_counts=True)
    num_k = min(1023, int(np.max(label_counts)))
    return {k:v for k, v in zip(unique_labels, label_counts)}, num_k

def get_lone_query_labels(query_labels, ref_labels, ref_label_counts, embeds_same_source):
    if embeds_same_source:
        return np.array([k for k, v in ref_label_counts.items() if v <= 1])
    else:
        return np.setdiff1d(query_labels, ref_labels)

def r_precision(knn_labels, gt_labels, embeds_same_source, label_counts):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels, embeds_same_source, label_counts)
    matches_per_row = np.sum((knn_labels == gt_labels) * relevance_mask.astype(bool), axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = matches_per_row / max_possible_matches_per_row
    return np.mean(accuracy_per_sample)

def mean_average_precision_at_r(knn_labels, gt_labels, embeds_same_source, label_counts):
    relevance_mask = get_relevance_mask(knn_labels.shape, gt_labels, embeds_same_source, label_counts)
    num_samples, num_k = knn_labels.shape
    equality = (knn_labels == gt_labels) * relevance_mask.astype(bool)
    cumulative_correct = np.cumsum(equality, axis=1)
    k_idx = np.tile(np.arange(1, num_k+1), (num_samples, 1))
    precision_at_ks = (cumulative_correct * equality) / k_idx
    summed_precision_pre_row = np.sum(precision_at_ks * relevance_mask, axis=1)
    max_possible_matches_per_row = np.sum(relevance_mask, axis=1)
    accuracy_per_sample = summed_precision_pre_row / max_possible_matches_per_row
    return np.mean(accuracy_per_sample)

def precision_at_k(knn_labels, gt_labels, k):
    curr_knn_labels = knn_labels[:, :k]
    accuracy_per_sample = np.sum(curr_knn_labels == gt_labels, axis=1) / k 
    return np.mean(accuracy_per_sample)

def recall_at_k(knn_labels, gt_labels, k):
    accuracy_per_sample = np.array([float(gt_label in recalled_predictions[:k]) for gt_label, recalled_predictions in zip(gt_labels, knn_labels)])
    return np.mean(accuracy_per_sample)

def f1_score(query_labels, cluster_labels):
    # compute tp_plus_fp
    qlabels_set, qlabels_counts = np.unique(query_labels, return_counts=True)
    tp_plut_fp = sum([comb(item, 2) for item in qlabels_counts if item > 1])

    # compute tp
    tp = sum([sum([comb(item, 2) for item in np.unique(cluster_labels[query_labels==query_label], return_counts=True)[1] if item > 1]) for query_label in qlabels_set])

    # compute fp
    fp = tp_plut_fp - tp

    # compute fn
    fn = sum([comb(item, 2) for item in np.unique(cluster_labels, return_counts=True)[1] if item > 1]) - tp

    # compute F1
    P, R = tp / (tp+fp), tp / (tp+fn)
    F1 = 2*P*R / (P+R)
    return F1