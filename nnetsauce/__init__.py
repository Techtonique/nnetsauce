"""
-- adapted from sklearn
"""
import sys
import logging
import os

from ._config import get_config, set_config, config_context

logger = logging.getLogger(__name__)


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.11.3"


# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded. It should not degrade performances since we manually
# take care of potential over-subcription performance issues, in sections of
# the code where nested OpenMP loops can happen, by dynamically reconfiguring
# the inner OpenMP runtime to temporarily disable it while under the scope of
# the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of nnetsauce when
    # the binaries are not built
    # mypy error: Cannot determine type of '__NNETSAUCE_SETUP__'
    __NNETSAUCE_SETUP__  # type: ignore
except NameError:
    __NNETSAUCE_SETUP__ = False

if __NNETSAUCE_SETUP__:

    sys.stderr.write("Partial import of nnetsauce during the build process.\n")
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet

else:

    from .base.base import Base
    from .base.baseRegressor import BaseRegressor
    from .boosting.adaBoostClassifier import AdaBoostClassifier
    from .custom.customClassifier import CustomClassifier
    from .custom.customRegressor import CustomRegressor
    from .glm.glmClassifier import GLMClassifier
    from .glm.glmRegressor import GLMRegressor
    from .mts.mts import MTS
    from .multitask.multitaskClassifier import MultitaskClassifier
    from .optimizers._optimizer import Optimizer
    from .randombag._randomBagClassifier import RandomBagClassifier
    from .randombag._randomBagRegressor import RandomBagRegressor
    from .ridge2.ridge2Classifier import Ridge2Classifier
    from .ridge2.ridge2Regressor import Ridge2Regressor
    from .ridge2.ridge2MultitaskClassifier import Ridge2MultitaskClassifier

    # from .rnn.rnnRegressor import RNNRegressor
    # from .rnn.rnnClassifier import RNNClassifier
    from .rvfl.bayesianrvflRegressor import BayesianRVFLRegressor
    from .rvfl.bayesianrvfl2Regressor import BayesianRVFL2Regressor
    from .simulator import Simulator

    __all__ = [
        "AdaBoostClassifier",
        "Base",
        "BaseRegressor",
        "BayesianRVFLRegressor",
        "BayesianRVFL2Regressor",
        "CustomClassifier",
        "CustomRegressor",
        "GLMRegressor",
        "GLMClassifier",
        "MTS",
        "MultitaskClassifier",
        "Optimizer",
        "RandomBagRegressor",
        "RandomBagClassifier",
        "Ridge2Regressor",
        "Ridge2Classifier",
        "Ridge2MultitaskClassifier",
        "Simulator",
        #    "RNNRegressor",
        #    "RNNClassifier",
    ]


def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    import os
    import numpy as np
    import random

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get("NNETSAUCE_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
