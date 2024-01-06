# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import pickle
import numpy as np
from ..utils import Progbar

# 0 - utils -----

# 1 main fitting loop -----


# For classification
def rbagloop_classification(base_learner, X, y, n_estimators, verbose, seed):
    voter = {}

    if verbose == 1:
        pbar = Progbar(n_estimators)

        for m in range(n_estimators):
            try:
                base_learner.set_params(seed=seed + m * 1000)

                base_learner.fit(np.asarray(X), np.asarray(y))

                voter[m] = pickle.loads(pickle.dumps(base_learner, -1))

                pbar.update(m)

            except Warning:
                pbar.update(m)

                continue

        pbar.update(n_estimators)

        return voter

    # verbose != 1:
    for m in range(n_estimators):
        try:
            base_learner.set_params(seed=seed + m * 1000)

            base_learner.fit(np.asarray(X), np.asarray(y))

            voter[m] = pickle.loads(pickle.dumps(base_learner, -1))

        except Warning:
            continue

    return voter


# For regression
def rbagloop_regression(base_learner, X, y, n_estimators, verbose, seed):
    voter = {}

    if verbose == 1:
        pbar = Progbar(n_estimators)

        for m in range(n_estimators):
            try:
                base_learner.set_params(seed=seed + m * 1000)

                base_learner.fit(np.asarray(X), np.asarray(y))

                voter[m] = pickle.loads(pickle.dumps(base_learner, -1))

                pbar.update(m)

            except Warning:
                pbar.update(m)

                continue

        pbar.update(n_estimators)

        return voter

    # verbose != 1:
    for m in range(n_estimators):
        try:
            base_learner.set_params(seed=seed + m * 1000)

            base_learner.fit(np.asarray(X), np.asarray(y))

            voter[m] = pickle.loads(pickle.dumps(base_learner, -1))

        except Warning:
            continue

    return voter
