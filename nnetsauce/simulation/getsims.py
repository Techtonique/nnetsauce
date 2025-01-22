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
        res = [[input_data[hx][ix][0][i] for i in range(n_sims)] for hx in range(h)]
        return np.asarray(res)


def getsimsxreg(sims_ix, output_dates, target_cols):
    """Get simulations from indices when using external regressors"""
    # Convert numpy array to list of DataFrames
    sims = tuple(
        pd.DataFrame(
            sims_ix[:, [i]],  # Keep 2D array shape with single column
            columns=[target_cols[i]],  # Use target column name
            index=output_dates,
        )
        for i in range(sims_ix.shape[1])  # Iterate over columns
    )

    return sims
