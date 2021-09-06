"""
This module defines different kinds of distance (or similarity) function.

+---------------+-----------------------+
| Type          | Supported metrics     |
+===============+=======================+
| Distance      | euclid, snr           |
+---------------+-----------------------+
| Similarity    | cosine, moco          |
+---------------+-----------------------+
"""

from .metric_factory import MetricFactory