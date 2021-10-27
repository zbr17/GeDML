"""
Collectors have two main functions: synthesizing (or collecting) samples and compute metric matrix (which will be passed to selectors and losses). 

All methods are listed below:

+-----------------------+-------------------------------------------------------------------------------+
| method                | description                                                                   |
+=======================+===============================================================================+
| BaseCollector         | Base class.                                                                   |
+-----------------------+-------------------------------------------------------------------------------+
| DefaultCollector      | Do nothing.                                                                   |
+-----------------------+-------------------------------------------------------------------------------+
| ProxyCollector        | Maintain a set of proxies                                                     |
+-----------------------+-------------------------------------------------------------------------------+
| MoCoCollector         | paper: **Momentum Contrast for Unsupervised Visual Representation Learning**  |
+-----------------------+-------------------------------------------------------------------------------+
| SimSiamCollector      | paper: **Exploring Simple Siamese Representation Learning**                   |
+-----------------------+-------------------------------------------------------------------------------+
| HDMLCollector         | paper: **Hardness-Aware Deep Metric Learning**                                |
+-----------------------+-------------------------------------------------------------------------------+
| DAMLCollector         | paper: **Deep Adversarial Metric Learning**                                   |
+-----------------------+-------------------------------------------------------------------------------+
| DVMLCollector         | paper: **Deep Variational Metric Learning**                                   |
+-----------------------+-------------------------------------------------------------------------------+

Notes:
    ``embedders`` have significent difference with ``collectors``. ``embedders`` also take charge of generating embeddings which will be used to compute metrics.

Todo:
    ``epoch-based collector``

"""

from .iteration_collectors import (
    DefaultCollector,
    ProxyCollector,
    MoCoCollector,
    SimSiamCollector,
    HDMLCollector,
    DAMLCollector,
    DVMLCollector
)

from .epoch_collectors import (
    GlobalProxyCollector,
    _DefaultGlobalCollector
)
from .base_collector import BaseCollector