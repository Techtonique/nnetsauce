#!/usr/bin/env python

"""
docstring
"""

# Authors: Henrik Linusson
# Yaniv Romano modified np.py file to include CQR
# T. Moudiki modified __init__.py to import classes

# __version__ = '2.1.0'

from .nc import (
    AbsErrorErrFunc,
    QuantileRegErrFunc,
    RegressorNc,
    RegressorNormalizer,
)
from .cp import IcpRegressor, TcpClassifier
from .icp import IcpClassifier
from .nc import ClassifierNc, MarginErrFunc
from .base import RegressorAdapter, ClassifierAdapter

__all__ = [
    "AbsErrorErrFunc",
    "MarginErrFunc",
    "QuantileRegErrFunc",
    "RegressorAdapter",
    "ClassifierAdapter",
    "RegressorNc",
    "ClassifierNc",
    "RegressorNormalizer",
    "IcpRegressor",
    "IcpClassifier",
    "TcpClassifier",
]
