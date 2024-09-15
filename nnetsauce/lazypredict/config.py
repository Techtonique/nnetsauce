from functools import partial
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import all_estimators
from ..multitask import MultitaskClassifier, SimpleMultitaskClassifier
from ..mts import ClassicalMTS


removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "HistGradientBoostingClassifier",
    "MultiOutputClassifier",
    "MultinomialNB",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "StackingClassifier",
    "VotingClassifier",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression",
    "CCA",
    "GaussianProcessRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "IsotonicRegression",
    "KernelRidge",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "NuSVR",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PLSCanonical",
    "PLSRegression",
    "RadiusNeighborsRegressor",
    "RegressorChain",
    "StackingRegressor",
    "SVR",
    "VotingRegressor",
]

CLASSIFIERS = [
    ("CustomClassifier(" + est[0] + ")", est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], ClassifierMixin)
        and (est[0] not in removed_classifiers)
    )
]

DEEPCLASSIFIERS = [
    ("DeepCustomClassifier(" + est[0] + ")", est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], ClassifierMixin)
        and (est[0] not in removed_classifiers)
    )
]

MULTITASKCLASSIFIERS = [
    (
        "MultitaskClassifier(" + est[0] + ")",
        partial(MultitaskClassifier, obj=est[1]()),
    )
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]

DEEPMULTITASKCLASSIFIERS = [
    (
        "DeepMultitaskClassifier(" + est[0] + ")",
        partial(MultitaskClassifier, obj=est[1]()),
    )
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]

SIMPLEMULTITASKCLASSIFIERS = [
    (
        "SimpleMultitaskClassifier(" + est[0] + ")",
        partial(SimpleMultitaskClassifier, obj=est[1]()),
    )
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]

DEEPSIMPLEMULTITASKCLASSIFIERS = [
    (
        "DeepSimpleMultitaskClassifier(" + est[0] + ")",
        partial(SimpleMultitaskClassifier, obj=est[1]()),
    )
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]

REGRESSORS = [
    ("CustomRegressor(" + est[0] + ")", est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]

DEEPREGRESSORS = [
    ("DeepCustomRegressor(" + est[0] + ")", est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]

REGRESSORSMTS = [
    ("MTS(" + est[0] + ")", est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]

DEEPREGRESSORSMTS = [
    ("DeepMTS(" + est[0] + ")", est[1])
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in removed_regressors)
    )
]

CLASSICALMTS = [
    ("ClassicalMTS(" + est[0] + ")", est[1])
    for est in [
        ("VAR", partial(ClassicalMTS, model="VAR")),
        ("VECM", partial(ClassicalMTS, model="VECM")),
    ]
]
