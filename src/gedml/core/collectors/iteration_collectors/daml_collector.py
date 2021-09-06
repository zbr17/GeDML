import torch
import torch.nn.functional as F

from ..base_collector import BaseCollector

class DAMLCollector(BaseCollector):
    """
    NOTE: only support Triplet-Loss.

    Paper: `Deep Adversarial Metric Learning <https://openaccess.thecvf.com/content_cvpr_2018/html/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.html>`_

    Training steps:

    1. pretrain the deep metric learning model without the hard negative generator;
    2. initialize the generator adversarial to the pre-trained metric;
    3. jointly optimize both networks during each iteration end-to-end

    Three losses for hard negative generation:

    1. the synthetic samples should be close to the anchor in the original feature space;
    2. the synthetic samples should perserve the annotation information;
    3. the synthetic samples should be misclassified by the learned metric

    Default backbone structure:

    1. trunk: ``GoogLeNet``
    2. embedder: one-layer perceptron
    3. generator: three-layer perceptron

    Args:
        embedder (torch.nn.Module): embedder model (default: one-layer perceptron)
        generator (torch.nn.Module): generator model (default: three-layer perceptron)
        lambda_0 (int): default: 1
        lambda_1 (int): default: 1
        lambda_2 (int): default: 50
        alpha (int): default: 1
    """
    def __init__(
        self,
        embedder,
        generator,
        lambda_0=1,
        lambda_1=1,
        lambda_2=50,
        alpha=1,
        *args,
        **kwargs
    ):
        super(DAMLCollector, self).__init__(*args, **kwargs)
        self.embedder = embedder
        self.generator = generator
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha

    def update(self, trainer):
        pass

    def forward(
        self,
        data,
        embeddings,
        features,
        labels
    ) -> tuple:
        """
        There are four losses to be computed in ``collect`` function (All losses will be computed in this function, i.e. NOT pass to ``selectors`` or ``losses`` modules)

        :math:`loss_{total} = \lambda_0 \\times loss_m + \lambda_1 \\times loss_{reg} + \lambda_2 \\times loss_{adv} + loss_{hard}`

        :math:`loss_m = mean(ReLU(D_{ap emb} - D_{an emb} - \\alpha))`

        :math:`loss_{adv} = mean(ReLU(D_{an feat} - D_{ap feat} - \\alpha))`

        :math:`loss_{reg} = mean(|f_{syn} - f_{neg}|^2_2)`

        :math:`loss_{hard} = mean(|f_{syn} - f_{anchor}|^2_2)`
        """
        # get a triplet
        matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).byte()
        diffs = matches ^ 1
        matches.fill_diagonal_(0)
        triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
        anc_idx, pos_idx, neg_idx = torch.where(triplets)

        anc_features = features[anc_idx]
        pos_features = features[pos_idx]
        neg_features = features[neg_idx]
        anc_embeddings = embeddings[anc_idx]
        pos_embedding = embeddings[pos_idx]

        # generate synthetic samples
        cat_features = torch.cat([anc_features, pos_features, neg_features], dim=-1)
        syn_features = self.generator(cat_features)
        syn_embeddings = self.embedder(syn_features)

        # compute loss_reg
        loss_reg = torch.mean(
            torch.sum((syn_features - neg_features)**2, dim=-1)
        )

        # compute loss_hard
        loss_hard = torch.mean(
            torch.sum((syn_features - anc_features)**2, dim=-1)
        )

        # compute loss_adv
        dist_an_feat = torch.sum(((anc_features - syn_features)**2), dim=-1)
        dist_ap_feat = torch.sum(((anc_features - pos_features)**2), dim=-1)
        loss_adv = torch.mean(
            F.relu(dist_an_feat - dist_ap_feat - self.alpha)
        )

        # compute loss_m
        dist_an_emb = torch.sum(((anc_embeddings - syn_embeddings)**2), dim=-1)
        dist_ap_emb = torch.sum(((anc_embeddings - pos_embedding)**2), dim=-1) 
        loss_m = torch.mean(
            F.relu(dist_ap_emb - dist_an_emb - self.alpha)
        )

        return (
            loss_m,
            self.lambda_0,
            loss_reg,
            self.lambda_1,
            loss_hard,
            1,
            loss_adv,
            self.lambda_2
        )

    