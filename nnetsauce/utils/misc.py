from math import floor
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

    n = len(y)

    for idx in range(n):
        if ((y[idx] - floor(y[idx])) != 0):            
            return False

    return True


# flatten list of lists
flatten = lambda l: [item for sublist in l for item in sublist]
