# Authors: Thierry Moudiki
#
# License: BSD 3


import numpy as np
from ..utils import misc as mx


# stratified subsampling
def subsample(y, row_sample=0.8, seed=123):

    assert (row_sample < 1) & (
        row_sample >= 0
    ), "'row_sample' must be < 1 and >= 0"
    n_obs = len(y)
    n_obs_out = np.ceil(n_obs * row_sample)

    # preproc -----
    if mx.is_factor(y):

        classes, n_elem_classes = np.unique(y, return_counts=True)
        n_classes = len(classes)
        y_as_classes = y.copy()
        freqs_hist = n_elem_classes / n_obs

    else:

        h = np.histogram(y, bins="auto")
        n_elem_classes = h[0]
        freqs_hist = n_elem_classes / n_obs
        breaks = h[1]

        n_breaks_1 = len(breaks) - 1
        classes = range(n_breaks_1)
        n_classes = n_breaks_1
        y_as_classes = np.asarray(
            [
                int(i) if ((y > breaks[i]) & (y <= breaks[i + 1])) else None
                for i in classes
            ]
        )

    # main loop ----
    index = []

    np.random.seed(seed)

    for i in range(n_classes):

        # index_class_i = [i for i, e in enumerate(bool_class_i) if e == True]
        index_class_i = np.where(y_as_classes == classes[i])[0]

        # at least 2 elements in class  #i
        # or only one element in class
        index.append(
            np.random.choice(
                index_class_i,
                size=int(n_obs_out * freqs_hist[i]),  # output size
                replace=True,
            ).tolist()
        ) if len(index_class_i) > 1 else index.append(index_class_i[0])

    try:

        return np.asarray(mx.flatten(index))

    except:

        return np.asarray(index)


# if __name__== "main":
#
#    import matplotlib.pyplot as plt
#    from scipy.stats import entropy
#    from collections import Counter

#    n_obs = 1000
#
#    y = np.random.rand(n_obs)
#    h = np.histogram(y, bins='auto')
#
#    y_factor = np.random.choice([0, 1], size = n_obs)
#    h_factor = np.histogram(y_factor, bins='auto')
#
#    # subsamples ----
#
#    ## continous
#    index_new = subsample(y, row_sample = 0.4)
#    y_new = y[index_new]
#    print(len(y))
#    print(len(y_new))
#
#    ## factor
#    index_new_factor = subsample(y_factor, row_sample = 0.4)
#    y_new_factor = y_factor[index_new_factor]
#    print(len(y_factor))
#    print(len(y_new_factor))
#
#    # graph 1
#    plt.hist(y, bins='auto', density=True)
#    plt.hist(y_new, bins=h[1], density=True)
#
#    # graph 2
#    plt.hist(y_factor, bins='auto', density=True)
#    plt.hist(y_new_factor, bins=h_factor[1], density=True)
#
#    # control
#    entropy(pk=h[0]/n_obs,
#            qk=np.histogram(y_new, bins=h[1])[0]/n_obs)
#
#    entropy(pk=h_factor[0]/n_obs,
#            qk=np.histogram(y_new_factor, bins=h_factor[1])[0]/n_obs)


# from sklearn.datasets import load_digits

# digits = load_digits()
# Z = digits.data
# t = digits.target
#
# Counter(t)
#
# index = rebalance(t)
# Counter(t[index])
#
# index2 = rebalance(t, down=False)
# Counter(t[index2])
