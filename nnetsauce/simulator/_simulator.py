
from . import _simulatorc as simulatorc


class Simulator():    

    def __init__(self, n_points, n_dims, skip=1):
        self.n = n_points            
        self.m = n_dims                
        self.skip = skip 

    def draw(self, type_sim="sobol"):
        h_sim = {
            "sobol": simulatorc.py_i4_sobol_generate
        }
        return h_sim[type_sim](self.m, self.n, self.skip)


