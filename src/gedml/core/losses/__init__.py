"""
All losses are divided into three groups:

1. **classifier-based**
2. **pair-based**
3. **proxy-based**

classifier-based
++++++++++++++++++++++++

+---------------------------+-----------------------------------------------------------------------------------+
| method                    | description                                                                       |
+===========================+===================================================================================+
| CrossEntropyLoss          | Cross entropy loss for unsupervised methods                                       |
+---------------------------+-----------------------------------------------------------------------------------+
| LargeMarginSoftmaxLoss    | paper: **Large-Margin Softmax Loss for Convolutional Neural Networks**            |
+---------------------------+-----------------------------------------------------------------------------------+
| ArcFaceLoss               | paper: **ArcFace: Additive Angular Margin Loss for Deep Face Recognition**        |
+---------------------------+-----------------------------------------------------------------------------------+
| CosFaceLoss               | paper: **CosFace: Large Margin Cosine Loss for Deep Face Recognition**            |
+---------------------------+-----------------------------------------------------------------------------------+

pair-based
++++++++++++++++++++++++

+---------------------------+-------------------------------------------------------------------------------------------------------+
| method                    | description                                                                                           |
+===========================+=======================================================================================================+
| ContrastiveLoss           | paper: **Learning a Similarity Metric Discriminatively, with Application to Face Verification**       |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| MarginLoss                | paper: **Sampling Matters in Deep Embedding Learning**                                                |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| TripletLoss               | paper: **Learning local feature descriptors with triplets and shallow convolutional neural networks** |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| AngularLoss               | paper: **Deep Metric Learning with Angular Loss**                                                     |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| CircleLoss                | paper: **Circle Loss: A Unified Perspective of Pair Similarity Optimization**                         |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| FastAPLoss                | paper: **Deep Metric Learning to Rank**                                                               |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| LiftedStructureLoss       | paper: **Deep Metric Learning via Lifted Structured Feature Embedding**                               |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| MultiSimilarityLoss       | paper: **Multi-Similarity Loss With General Pair Weighting for Deep Metric Learning**                 |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| NPairLoss                 | paper: **Improved Deep Metric Learning with Multi-class N-pair Loss Objective**                       |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| SignalToNoiseRatioLoss    | paper: **Signal-To-Noise Ratio: A Robust Distance Metric for Deep Metric Learning**                   |
+---------------------------+-------------------------------------------------------------------------------------------------------+
| PosPairLoss               | paper: **Exploring Simple Siamese Representation Learning**                                           |
+---------------------------+-------------------------------------------------------------------------------------------------------+

proxy-based
++++++++++++++++++++++++

+-------------------+---------------------------------------------------------------------------+
| method            | description                                                               |
+===================+===========================================================================+
| ProxyLoss         | paper: **No Fuss Distance Metric Learning Using Proxies**                 |
+-------------------+---------------------------------------------------------------------------+
| ProxyAnchorLoss   | paper: **Proxy Anchor Loss for Deep Metric Learning**                     |
+-------------------+---------------------------------------------------------------------------+
| SoftTripleLoss    | paper: **SoftTriple Loss: Deep Metric Learning Without Triplet Sampling** |
+-------------------+---------------------------------------------------------------------------+

Example:
    Take ``ContrastiveLoss`` for example:

    >>> loss_func = ContrastiveLoss()
    >>> x = torch.randn(100, 128)
    >>> mat = torch.cdist(x, x)
    >>> labels = torch.randint(0, 10, size=(100,))
    >>> loss = loss_func(mat, labels.unsqueeze(1), labels.unsqueeze(0))
"""

from .base_loss import BaseLoss

from .classifier_based_loss import (
    CrossEntropyLoss,
    LargeMarginSoftmaxLoss,
    ArcFaceLoss,
    CosFaceLoss
)

from .pair_based_loss import (
    ContrastiveLoss,
    MarginLoss,
    TripletLoss,
    AngularLoss,
    CircleLoss,
    FastAPLoss,
    LiftedStructureLoss,
    MultiSimilarityLoss,
    NPairLoss,
    SignalToNoiseRatioLoss,
    PosPairLoss
)

from .proxy_based_loss import (
    ProxyLoss,
    ProxyAnchorLoss,
    SoftTripleLoss
)