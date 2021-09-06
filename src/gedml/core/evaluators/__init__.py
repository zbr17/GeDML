"""
This module implements the evaluation indicators, such as F1-score, recall@k, precision@k and etc using faiss-gpu package.

- ``calculator.py``: A evaluation manager class.
- ``metrics.py``: All evaluation functions are implemented here.
"""

from .calculator import Calculator
from .calculator_mat import CalculatorFromMat