# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from ..utils.misc import flatten, is_factor


def dosubsample(y, row_sample=0.8, seed=123, n_jobs=None, verbose=False):
    index = []
    assert (row_sample < 1) & (
        row_sample >= 0
    ), "'row_sample' must be < 1 and >= 0"
    n_obs = len(y)
    n_obs_out = np.ceil(n_obs * row_sample)

    # preproc -----
    if is_factor(y):  # classification

        classes, n_elem_classes = np.unique(y, return_counts=True)
        n_classes = len(classes)
        y_as_classes = y.copy()
        freqs_hist = np.zeros_like(n_elem_classes, dtype=float)
        if verbose is True:
            print(f"creating breaks...")
        if n_jobs is None:
            for i in range(len(n_elem_classes)):
                freqs_hist[i] = float(n_elem_classes[i]) / n_obs
        else:

            def get_freqs_hist(i):
                return float(n_elem_classes[i]) / n_obs

            freqs_hist = Parallel(n_jobs=n_jobs)(
                delayed(get_freqs_hist)(i) for i in range(len(n_elem_classes))
            )

    else:  # regression

        h = np.histogram(y, bins="auto")
        n_elem_classes = np.asarray(h[0], dtype=np.int32)
        freqs_hist = np.zeros_like(n_elem_classes, dtype=float)

        if verbose is True:
            print(f"creating breaks...")
        if n_jobs is None:
            for i in range(len(n_elem_classes)):
                freqs_hist[i] = float(n_elem_classes[i]) / n_obs
        else:

            def get_freqs_hist(i):
                return float(n_elem_classes[i]) / n_obs

            freqs_hist = Parallel(n_jobs=n_jobs)(
                delayed(get_freqs_hist)(i) for i in range(len(n_elem_classes))
            )

        breaks = h[1]
        n_breaks_1 = len(breaks) - 1
        classes = range(n_breaks_1)
        n_classes = n_breaks_1
        y_as_classes = np.zeros_like(y, dtype=int)

        for i in classes:
            y_as_classes[(y > breaks[i]) * (y <= breaks[i + 1])] = int(i)

    # main loop ----

    if verbose is True:
        print(f"main loop...")

    if n_jobs is None:

        if verbose is True:
            iterator = tqdm(range(n_classes))
        else:
            iterator = range(n_classes)

        for i in iterator:
            bool_class_i = y_as_classes == classes[i]
            index_class_i = np.asarray(
                np.where(bool_class_i == True)[0], dtype=np.int32
            )
            if np.sum(bool_class_i) > 1:  # at least 2 elements in class  #i
                np.random.seed(seed + i)
                index.extend(
                    np.random.choice(
                        index_class_i,
                        size=int(n_obs_out * freqs_hist[i]),  # output size
                        replace=True,
                    ).tolist()
                )
            else:  # only one element in class
                try:
                    index.append(index_class_i[0])
                except:
                    pass

    else:  # parallel execution

        def get_index(i):
            bool_class_i = y_as_classes == classes[i]
            index_class_i = np.asarray(
                np.where(bool_class_i == True)[0], dtype=np.int32
            )
            if np.sum(bool_class_i) > 1:  # at least 2 elements in class  #i
                np.random.seed(seed + i)
                return np.random.choice(
                    index_class_i,
                    size=int(n_obs_out * freqs_hist[i]),  # output size
                    replace=True,
                ).tolist()
            else:  # only one element in class
                try:
                    return index_class_i[0]
                except:
                    pass

        if verbose is True:
            index = Parallel(n_jobs=n_jobs)(
                delayed(get_index)(i) for i in tqdm(range(n_classes))
            )
        else:
            index = Parallel(n_jobs=n_jobs)(
                delayed(get_index)(i) for i in range(n_classes)
            )

    try:
        return np.asarray(flatten(index))
    except:
        return np.asarray(index)
