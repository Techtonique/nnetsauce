from functools import partial
from . import _simulatorc as simulatorc
from ..utils import memoize
from ..simulation import generate_uniform


class Simulator:
    def __init__(self, n_points, n_dims, type_sim="sobol", seed=123):

        self.n = n_points
        self.m = n_dims
        self.type_sim = type_sim
        self.seed = seed

    def draw(self):
        h_sim = {
            "sobol": simulatorc.py_i4_sobol_generate,
            "halton": simulatorc.py_halton_sequence,
            "hammersley": simulatorc.py_hammersley_sequence,
            # "uniform": partial(generate_uniform, seed=self.seed),
        }
        return h_sim[self.type_sim](m=self.m, n=self.n)
