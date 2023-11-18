
from functools import partial 
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.discovery import all_estimators
from ..multitask import MultitaskClassifier, SimpleMultitaskClassifier


removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor", 
]

CLASSIFIERS = [
    ("Custom" + est[0], est[1])
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]

MULTITASKCLASSIFIERS = [
    ("Multitask" + est[0], partial(MultitaskClassifier, obj=est[1]()))
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

SIMPLEMULTITASKCLASSIFIERS = [
    ("SimpleMultitask" + est[0], partial(SimpleMultitaskClassifier, obj=est[1]()))
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

REGRESSORS = [
    ("Custom" + est[0], est[1])
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]
