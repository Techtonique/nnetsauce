import numpy as np
import pandas as pd


def getsims(input_data, ix):

    if isinstance(input_data[0], pd.DataFrame):  # kde
        n_sims = len(input_data)
        res = [input_data[i].iloc[:, ix].values for i in range(n_sims)]
        return np.asarray(res).T

    if isinstance(input_data[0], tuple):  # GP posterior
        h = len(input_data)
        n_sims = len(input_data[0][0][0])
        res = [
            [input_data[hx][ix][0][i] for i in range(n_sims)] for hx in range(h)
        ]
        return np.asarray(res)
