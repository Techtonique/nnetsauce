import numpy as np 

def getsims(input_tuple, ix):
    n_sims = len(input_tuple)
    res = [input_tuple[i][:, ix] for i in range(n_sims)]
    return np.asarray(res).T