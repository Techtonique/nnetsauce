from .base.base import Base
from .base.baseRegressor import BaseRegressor
from .boosting.adaBoostClassifier import AdaBoostClassifier
from .custom.customClassifier import CustomClassifier
from .custom.customRegressor import CustomRegressor
from .multitask.multitaskClassifier import MultitaskClassifier
from .mts import MTS
from .randombag.randomBagClassifier import RandomBagClassifier
from .ridge2.ridge2Classifier import Ridge2Classifier
from .ridge2.ridge2Regressor import Ridge2Regressor
from .ridge2.ridge2MultitaskClassifier import Ridge2MultitaskClassifier

# from .rnn.rnnRegressor import RNNRegressor
# from .rnn.rnnClassifier import RNNClassifier
from .rvfl.bayesianrvflRegressor import BayesianRVFLRegressor
from .rvfl.bayesianrvfl2Regressor import BayesianRVFL2Regressor


__all__ = [
    "AdaBoostClassifier",
    "Base",
    "BaseRegressor",
    "BayesianRVFLRegressor",
    "BayesianRVFL2Regressor",
    "CustomClassifier",
    "CustomRegressor",
    "MultitaskClassifier",
    "RandomBagClassifier",
    "Ridge2Regressor",
    "Ridge2Classifier",
    "Ridge2MultitaskClassifier",
    #    "RNNRegressor",
    #    "RNNClassifier",
    "MTS",
]
