import numpy as np
import pandas as pd 

def getsims(input_data, ix):

    if isinstance(input_data[0], pd.DataFrame): # kde
        n_sims = len(input_data)
        res = [input_data[i].iloc[:, ix].values for i in range(n_sims)] 
        return np.asarray(res).T       

    if isinstance(input_data[0], tuple): # GP posterior
        print(f"\n\n input_data: {input_data} \n\n")
        h = len(input_data)
        print(f"\n\n h: {h} \n\n")
        n_sims = len(input_data[0][0][0])
        print(f"\n\n n_sims: {n_sims} \n\n")
        print(f"\n\n input_data[0][0]: {input_data[0][0]} \n\n")
        res = [[input_data[hx][ix][0][i] for i in range(n_sims)] for hx in range(h)]
        return np.asarray(res)

    