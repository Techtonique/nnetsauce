import numpy as np
from .memoize import memoize


# merge two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# check if x is int
@memoize
def is_int(x):
    try:
        return int(x) == x
    except:
        return False


# check if x is float
@memoize
def is_float(x):
    return isinstance(x, float)


# check if the response contains only integers
@memoize
def is_factor(y):
    return not np.mod(y, 1).any()


# flatten list of lists
flatten = lambda l: [item for sublist in l for item in sublist]
