import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from ..base_collector import BaseCollector

class HDMLCollector(BaseCollector):
    """
    Use variational autoencoder to decompose intra-class invariance and intra-class variance.

    Paper: `Hardness-Aware Deep Metric Learning <https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Hardness-Aware_Deep_Metric_Learning_CVPR_2019_paper.html>`_

    Four types of loss: 
    (loss_avg = loss_m,
    loss_gen = loss_recon + loss_soft)
    
    1. loss_recon
    2. loss_soft
    3. loss_syn
    4. loss_m

    Args:
        generator (torch.nn.Module): multi-layer perceptron
        embedder (torch.nn.Module): multi-layer perceptron
        classifier (torch.nn.Module): multi-layer perceptrons
        alpha (float): 90.0 (NPairLoss) or 7.0 (TripletLoss)
        beta (float): 1.0e4
        coef_lambda (float): 0.5
        soft_weight (float): 1.0e4
        d_plus_scheme (str): default: ``positive_distance``
        d_plus (float): Constant or positive pair distance. default: 0.5
    """
    def __init__(
        self,
        generator,
        embedder,
        classifier,
        alpha=90.0,
        beta=1.0e4,
        coef_lambda=0.5,
        soft_weight=1.0e4,
        d_plus_scheme="positive_distance",
        d_plus=0.5,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.coef_lambda = coef_lambda
        self.soft_weight = soft_weight
        self.d_plus_scheme = d_plus_scheme # TODO: 
        self.d_plus = d_plus

        self.generator = generator
        self.embedder = embedder
        self.classifier = classifier

        self.lambda_0 = None
        self.loss_avg = None # adjust the hardness
        self.loss_gen = None # adjust the 
    
    
    def update(self, trainer):
        """
        In HDML paper, an adaptive weighting method is proposed. Therefore, before each epoch ``loss_avg`` and ``loss_gen`` must be updated from outside ``trainer``

        :math:`loss_{avg} = loss_m`

        :math:`loss_{gen} = loss_{recon} + loss_{soft}`
        """
        loss_handler = trainer.loss_handler
        self.loss_avg = float(loss_handler.get("loss_m", self._default_loss_value, is_avg_value=True))
        self.loss_gen = float(
            loss_handler.get("loss_recon", self._default_loss_value, is_avg_value=True) + 
            loss_handler.get("loss_soft", self._default_loss_value, is_avg_value=True)
        )
        if not torch.is_tensor(self.loss_avg):
            self.loss_avg = torch.tensor(self.loss_avg)
        if not torch.is_tensor(self.loss_gen):
            self.loss_gen = torch.tensor(self.loss_gen)
    
    def construct_neg_embedding_hat(self, pos_embedding, neg_embedding):
        # compute lambda_0
        np_dist = F.pairwise_distance(pos_embedding, neg_embedding, p=2)
        d_plus = torch.ones_like(np_dist, device=np_dist.device) * self.d_plus
        self.lambda_loss = torch.exp( - self.alpha / self.loss_avg)
        self.lambda_0 = (
            self.lambda_loss + ( 1 - self.lambda_loss ) 
            * d_plus / np_dist
        ).unsqueeze(-1).detach()
        self.lambda_0[np_dist <= d_plus] = 1

        # construct embedding_hat
        neg_embedding_hat = (
            pos_embedding + 
            self.lambda_0 * (
                neg_embedding - pos_embedding
            )
        )
        
        return pos_embedding, neg_embedding_hat

    def generate_feature(self, embeddings):
        # generate features
        return self.generator(embeddings)
    
    def forward(
        self, 
        data, 
        embeddings, 
        features,
        labels
    ) -> tuple:
        """
        Define four kinds of losses.

        :math:`loss_{total} = w_{recon} \\times loss_{recon} + w_{soft} \\times loss_{soft} + w_m \\times loss_m + w_{syn} \\times loss_{syn}`

        :math:`loss_{recon} = mean(|f_{pos} - f_{pos-recon}|^2_2)`

        :math:`loss_{soft} = CrossEntropy(Prob_{recon}, Labels_{recon})`

        :math:`loss_m = loss_{metric}(matrix_{m})`

        :math:`loss_syn = loss_{metric}(matrix_{syn})`
        """
        batch_size = embeddings.size(0)
        # sample pos_idx and neg_idx
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).byte()
        # pos_idx, neg_idx = torch.where(neg_mask)
        pos_idx, neg_idx = [], []
        for i in range(batch_size):
            neg_list = torch.where(neg_mask[i])[0]
            if len(neg_list) > 0:
                pos_idx.append(i)
                neg_idx.append(np.random.choice(neg_list.cpu().numpy()))
        pos_idx = torch.tensor(pos_idx)
        neg_idx = torch.tensor(neg_idx)

        # get pos and neg pairs
        pos_embedding, neg_embedding = embeddings[pos_idx], embeddings[neg_idx]
        pos_labels, neg_labels = labels[pos_idx], labels[neg_idx]

        # construct hard negative embeddings
        pos_embedding, neg_embedding = self.construct_neg_embedding_hat(
            pos_embedding,
            neg_embedding
        )

        # generate hard negative features
        pos_recon_features = self.generate_feature(pos_embedding)
        neg_recon_features = self.generate_feature(neg_embedding)

        # compute reconstruction loss (loss_recon)
        loss_recon = torch.mean(torch.sum((features[pos_idx] - pos_recon_features).pow(2), dim=-1))
        weight_recon = self.coef_lambda

        # compute loss_m
        metric_mat_m = self.metric(embeddings, embeddings)
        row_labels_m = labels.unsqueeze(1)
        col_labels_m = labels.unsqueeze(0)
        is_same_source = True

        # compute loss_syn
        recon_features = torch.cat([pos_recon_features, neg_recon_features], dim=0)
        recon_labels = torch.cat([pos_labels, neg_labels], dim=0)
        metric_mat_syn = self.metric(recon_features, recon_features)
        row_labels_syn = recon_labels.unsqueeze(1)
        col_labels_syn = recon_labels.unsqueeze(0)

        # compute loss_soft
        recon_prob = self.classifier(recon_features)
        loss_soft = F.cross_entropy(recon_prob, recon_labels, reduction="mean") * self.coef_lambda
        weight_soft = self.coef_lambda * self.soft_weight
        
        # weight of loss_m and loss_syn
        weight_m = torch.exp( - self.beta / self.loss_gen)
        weight_syn = 1 - weight_m

        return (
            metric_mat_m,
            row_labels_m,
            col_labels_m,
            metric_mat_syn,
            row_labels_syn,
            col_labels_syn,
            is_same_source,
            loss_recon,
            loss_soft,
            weight_recon,
            weight_soft,
            weight_m,
            weight_syn
        )
    