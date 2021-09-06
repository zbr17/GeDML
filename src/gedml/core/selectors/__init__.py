"""
This module is aimed at **How to select samples to provide more information for training**.

+---------------------------+-----------------------------------------------+
| method                    | description                                   |
+===========================+===============================================+
| BaseSelector              | Base class.                                   |
+---------------------------+-----------------------------------------------+
| DefaultSelector           | Do nothing.                                   |
+---------------------------+-----------------------------------------------+
| DenseTripletSelector      | Select all triples.                           |
+---------------------------+-----------------------------------------------+
| DensePairSelector         | Select all pairs.                             |
+---------------------------+-----------------------------------------------+
| DistanceWeightedSelector  | Distance weighted selector                    |
+---------------------------+-----------------------------------------------+
| SemiHardSelector          | Semi-hard selector                            |
+---------------------------+-----------------------------------------------+
| RandomTripletSelector     | Randomly select triplets                      |
+---------------------------+-----------------------------------------------+
| HardSelector              | Hardest hard sample                           |
+---------------------------+-----------------------------------------------+
"""

from .base_selector import BaseSelector
from .default_selector import DefaultSelector
from .dense_triplet_selector import DenseTripletSelector
from .dense_pair_selector import DensePairSelector
from .distance_weighted_selector import DistanceWeightedSelector
from .semi_hard_selector import SemiHardSelector
from .random_triplet_selector import RandomTripletSelector
from .hard_selector import HardSelector
from .hard_pair_selector import HardPairSelector