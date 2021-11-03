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

from .default_collector import DefaultCollector
from .proxy_collector import ProxyCollector
from .moco_collector import MoCoCollector
# from .simsiam_collector import SimSiamCollector
# from .hdml_collector import HDMLCollector
# from .daml_collector import DAMLCollector
# from .dvml_collector import DVMLCollector
from ._default_global_collector import _DefaultGlobalCollector
from .base_collector import BaseCollector