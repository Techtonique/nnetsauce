# Authors: Thierry Moudiki
#
# License: BSD 3 Clear Clause

import copy
import numpy as np
import pandas as pd
import sklearn.metrics as skm2
import matplotlib.pyplot as plt
from collections import namedtuple
from copy import deepcopy
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from ..base import Base
from ..mts import MTS
from ..conformal import SPCI_and_EnbPI
from ..simulation import getsims
from ..utils import matrixops as mo
from ..utils import timeseries as ts
from .deepRegressor import DeepRegressor


class DeepMTS(MTS):
    """Deep Univariate and multivariate time series (MTS) forecasting with Quasi-Randomized networks (Work in progress /!\)

    Parameters:

        obj: object.
            any object containing a method fit (obj.fit()) and a method predict
            (obj.predict()).

        n_hidden_features: int.
            number of nodes in the hidden layer.

        activation_name: str.
            activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'.

        a: float.
            hyperparameter for 'prelu' or 'elu' activation function.

        nodes_sim: str.
            type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
            'uniform'.

        bias: boolean.
            indicates if the hidden layer contains a bias term (True) or not
            (False).

        dropout: float.
            regularization parameter; (random) percentage of nodes dropped out
            of the training.

        direct_link: boolean.
            indicates if the original predictors are included (True) in model's fitting or not (False).

        n_clusters: int.
            number of clusters for 'kmeans' or 'gmm' clustering (could be 0: no clustering).

        cluster_encode: bool.
            defines how the variable containing clusters is treated (default is one-hot)
            if `False`, then labels are used, without one-hot encoding.

        type_clust: str.
            type of clustering method: currently k-means ('kmeans') or Gaussian
            Mixture Model ('gmm').

        type_scaling: a tuple of 3 strings.
            scaling methods for inputs, hidden layer, and clustering respectively
            (and when relevant).
            Currently available: standardization ('std') or MinMax scaling ('minmax').

        lags: int.
            number of lags used for each time series.

        replications: int.
            number of replications (if needed, for predictive simulation). Default is 'None'.

        kernel: str.
            the kernel to use for residuals density estimation (used for predictive simulation). Currently, either 'gaussian' or 'tophat'.

        agg: str.
            either "mean" or "median" for simulation of bootstrap aggregating

        seed: int.
            reproducibility seed for nodes_sim=='uniform' or predictive simulation.

        backend: str.
            "cpu" or "gpu" or "tpu".

        verbose: int.
            0: not printing; 1: printing

        show_progress: bool.
            True: progress bar when fitting each series; False: no progress bar when fitting each series

    Attributes:

        fit_objs_: dict
            objects adjusted to each individual time series

        y_: {array-like}
            MTS responses (most recent observations first)

        X_: {array-like}
            MTS lags

        xreg_: {array-like}
            external regressors

        y_means_: dict
            a dictionary of each series mean values

        preds_: {array-like}
            successive model predictions

        preds_std_: {array-like}
            standard deviation around the predictions

        return_std_: boolean
            return uncertainty or not (set in predict)

        df_: data frame
            the input data frame, in case a data.frame is provided to `fit`

    Examples:

    Example 1:

    ```python
    import nnetsauce as ns
    import numpy as np
    from sklearn import linear_model
    np.random.seed(123)

    M = np.random.rand(10, 3)
    M[:,0] = 10*M[:,0]
    M[:,2] = 25*M[:,2]
    print(M)

    # Adjust Bayesian Ridge
    regr4 = linear_model.BayesianRidge()
    obj_DeepMTS = ns.DeepMTS(regr4, lags = 1, n_hidden_features=5)
    obj_DeepMTS.fit(M)
    print(obj_DeepMTS.predict())

    # with credible intervals
    print(obj_DeepMTS.predict(return_std=True, level=80))

    print(obj_DeepMTS.predict(return_std=True, level=95))
    ```

    Example 2:

    ```python
    import nnetsauce as ns
    import numpy as np
    from sklearn import linear_model

    dataset = {
    'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
    'series1' : [34, 30, 35.6, 33.3, 38.1],
    'series2' : [4, 5.5, 5.6, 6.3, 5.1],
    'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
    df = pd.DataFrame(dataset).set_index('date')
    print(df)

    # Adjust Bayesian Ridge
    regr5 = linear_model.BayesianRidge()
    obj_DeepMTS = ns.DeepMTS(regr5, lags = 1, n_hidden_features=5)
    obj_DeepMTS.fit(df)
    print(obj_DeepMTS.predict())

    # with credible intervals
    print(obj_DeepMTS.predict(return_std=True, level=80))

    print(obj_DeepMTS.predict(return_std=True, level=95))
    ```
    """

    # construct the object -----

    def __init__(
        self,
        obj,
        verbose=0,
        # Defining depth
        n_layers=3,
        n_hidden_features=5,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=True,
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        lags=1,
        type_pi="kde",
        replications=None,
        kernel=None,
        agg="mean",
        seed=123,
        backend="cpu",
        show_progress=True,
    ):
        assert int(lags) == lags, "parameter 'lags' should be an integer"

        self.deepobj = DeepRegressor(
            obj=obj,
            verbose=0, # keep this at 0
            # Defining depth
            n_layers=n_layers,
            # CustomRegressor attributes
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
            nodes_sim=nodes_sim,
            bias=bias,
            dropout=dropout,
            direct_link=direct_link,
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            seed=seed,
            backend=backend,
        )

        super().__init__(
            obj=self.deepobj,
            n_hidden_features=n_hidden_features,
            activation_name=activation_name,
            a=a,
            nodes_sim=nodes_sim,
            bias=bias,
            dropout=dropout,
            direct_link=direct_link,
            n_clusters=n_clusters,
            cluster_encode=cluster_encode,
            type_clust=type_clust,
            type_scaling=type_scaling,
            seed=seed,
            backend=backend,
        )

        self.n_layers = n_layers        
        self.n_series = None
        self.lags = lags
        self.type_pi = type_pi
        self.replications = replications
        self.kernel = kernel
        self.agg = agg
        self.verbose = verbose
        self.show_progress = show_progress
        self.series_names = None
        self.input_dates = None        
        self.fit_objs_ = {}
        self.y_ = None  # MTS responses (most recent observations first)
        self.X_ = None  # MTS lags
        self.xreg_ = None
        self.y_means_ = {}
        self.mean_ = None
        self.upper_ = None
        self.lower_ = None
        self.output_dates_ = None
        self.preds_std_ = []
        self.alpha_ = None
        self.return_std_ = None
        self.df_ = None
        self.residuals_ = []
        self.residuals_sims_ = None
        self.kde_ = None
        self.sims_ = None