import numpy as np


# merge two dictionaries
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


# check if the response contains only integers
def is_factor(y):

    n = len(y)

    cond0 = (
        sum(
            list(
                map(
                    lambda x: isinstance(y[x], int),
                    range(n),
                )
            )
        )
        == n
    )
    cond1 = (
        sum(
            list(
                map(
                    lambda x: isinstance(y[x], np.int),
                    range(n),
                )
            )
        )
        == n
    )
    cond2 = (
        sum(
            list(
                map(
                    lambda x: isinstance(y[x], np.int32),
                    range(n),
                )
            )
        )
        == n
    )
    cond3 = (
        sum(
            list(
                map(
                    lambda x: isinstance(y[x], np.int64),
                    range(n),
                )
            )
        )
        == n
    )

    return cond0 or cond1 or cond2 or cond3


# flatten list of lists
flatten = lambda l: [
    item for sublist in l for item in sublist
]
