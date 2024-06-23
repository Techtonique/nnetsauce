import numpy as np


def test_check(x, y):
    return np.allclose(np.around(x, 2), y)
