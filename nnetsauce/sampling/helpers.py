
# Authors: Thierry Moudiki
#
# License: BSD 3 Clear

import numpy as np
from ..utils.misc import flatten, is_factor

def dosubsample(y, row_sample=0.8, seed=123):
        
    index = []

    assert (row_sample < 1) & (
        row_sample >= 0
    ), "'row_sample' must be < 1 and >= 0"

    n_obs = len(y)
    n_obs_out = np.ceil(n_obs * row_sample)

    # preproc -----
    if is_factor(y): # classification

        classes, n_elem_classes = np.unique(y, return_counts=True)
        n_classes = len(classes)
        y_as_classes = y.copy()
        freqs_hist = np.zeros_like(n_elem_classes, dtype=float)
        
        for i in range(len(n_elem_classes)):
            freqs_hist[i] = float(n_elem_classes[i]) / n_obs

    else: # regression

        h = np.histogram(y, bins="auto")
        n_elem_classes = np.asarray(h[0], dtype=np.integer)
        freqs_hist = np.zeros_like(n_elem_classes, dtype=float)
        
        for i in range(len(n_elem_classes)):
            freqs_hist[i] = float(n_elem_classes[i]) / n_obs
        
        breaks = h[1]

        n_breaks_1 = len(breaks) - 1
        classes = range(n_breaks_1)
        n_classes = n_breaks_1
        y_as_classes = np.zeros_like(y, dtype=int)

        for i in classes:
            y_as_classes[(y > breaks[i]) * (y <= breaks[i + 1])] = int(i)

    # main loop ----

    np.random.seed(seed)

    for i in range(n_classes):

        bool_class_i = (y_as_classes == classes[i])               

        # index_class_i = [i for i, e in enumerate(bool_class_i) if e == True]
        index_class_i = np.asarray(np.where(bool_class_i == True)[0], dtype=np.integer)

        if np.sum(bool_class_i) > 1:  # at least 2 elements in class  #i

            index.append(
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

                0

    try:
        return np.asarray(flatten(index))
    except:
        return np.asarray(index)
    