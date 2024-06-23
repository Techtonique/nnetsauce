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
from .cp import IcpRegressor
from .base import RegressorAdapter

__all__ = [
    "AbsErrorErrFunc",
    "QuantileRegErrFunc",
    "RegressorAdapter",
    "RegressorNc",
    "RegressorNormalizer",
    "IcpRegressor",
]
