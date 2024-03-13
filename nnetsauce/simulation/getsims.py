import numpy as np


def getsims(input_data, ix):
    try: 
        n_sims = len(input_data)
        res = [input_data[i].iloc[:, ix].values for i in range(n_sims)]        
    except: 
        print(f"\n\n input_data: {input_data} \n\n")
        n_sims = len(input_data[0])
        res = [input_data[ix][i] for i in range(n_sims)]
    return np.asarray(res).T