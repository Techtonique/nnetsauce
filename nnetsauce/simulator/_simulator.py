
from . import _simulatorc as simulatorc
from ..utils import memoize

class Simulator():    

    def __init__(self, n_points, n_dims, type_sim="sobol"):

        self.n = n_points            
        self.m = n_dims                
        self.type_sim = type_sim 

    @memoize
    def draw(self):
        h_sim = {
            "sobol": simulatorc.py_i4_sobol_generate,
            "halton": simulatorc.py_halton_sequence,
            "hammersley": simulatorc.py_hammersley_sequence
        }    
        return h_sim[self.type_sim](m=self.m, n=self.n)    
        

