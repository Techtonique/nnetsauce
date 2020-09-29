
from . import _simulatorc as simulatorc


class Simulator():    

    def __init__(self, n_points, n_dims, skip=1):
        self.n = n_points            
        self.m = n_dims                
        self.skip = skip 

    def draw(self):
        return simulatorc.py_i4_sobol_generate(self.m, self.n, self.skip)


