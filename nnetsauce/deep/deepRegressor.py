import numpy as np
import pandas as pd
import sklearn.metrics as skm2
from copy import deepcopy
from tqdm import tqdm
from sklearn.base import RegressorMixin
from ..custom import CustomRegressor
from ..utils import matrixops as mo


class DeepRegressor(CustomRegressor, RegressorMixin):
    """
    Deep Regressor

    Parameters:

        verbose : int, optional (default=0)
            Monitor progress when fitting.

    Examples:

        ```python
        import nnetsauce as ns
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import RidgeCV
        data = load_diabetes()
        X = data.data
        y= data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)
        obj = RidgeCV()
        clf = ns.DeepRegressor(obj)
        clf.fit(X_train, y_train)
        print(clf.score(clf.predict(X_test), y_test))
        ```

    """

    def __init__(
        self,
        obj,
        verbose=0,
        # Defining depth
        n_layers=3,
        # CustomRegressor attributes
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
        col_sample=1,
        row_sample=1,
        seed=123,
        backend="cpu",
    ):
        super().__init__(
            obj=obj,
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
            col_sample=col_sample,
            row_sample=row_sample,
            seed=seed,
            backend=backend,
        )

        assert n_layers >= 2, "must have n_layers >= 2"

        self.stacked_obj = obj
        self.verbose = verbose
        self.n_layers = n_layers

    def fit(self, X, y):
        """Fit Regression algorithms to X and y.
        Parameters
        ----------
        X : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        A fitted object
        """

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # init layer
        self.stacked_obj = CustomRegressor(
            obj=self.stacked_obj,
            n_hidden_features=self.n_hidden_features,
            activation_name=self.activation_name,
            a=self.a,
            nodes_sim=self.nodes_sim,
            bias=self.bias,
            dropout=self.dropout,
            direct_link=self.direct_link,
            n_clusters=self.n_clusters,
            cluster_encode=self.cluster_encode,
            type_clust=self.type_clust,
            type_scaling=self.type_scaling,
            col_sample=self.col_sample,
            row_sample=self.row_sample,
            seed=self.seed,
            backend=self.backend,
        )

        # self.stacked_obj.fit(X, y)

        if self.verbose > 0:
            iterator = tqdm(range(self.n_layers - 1))
        else:
            iterator = range(self.n_layers - 1)

        for _ in iterator:
            self.stacked_obj = deepcopy(
                CustomRegressor(
                    obj=self.stacked_obj,
                    n_hidden_features=self.n_hidden_features,
                    activation_name=self.activation_name,
                    a=self.a,
                    nodes_sim=self.nodes_sim,
                    bias=self.bias,
                    dropout=self.dropout,
                    direct_link=self.direct_link,
                    n_clusters=self.n_clusters,
                    cluster_encode=self.cluster_encode,
                    type_clust=self.type_clust,
                    type_scaling=self.type_scaling,
                    col_sample=self.col_sample,
                    row_sample=self.row_sample,
                    seed=self.seed,
                    backend=self.backend,
                )
            )

            # self.stacked_obj.fit(X, y)

        self.stacked_obj.fit(X, y)

        self.obj = deepcopy(self.stacked_obj)

        return self.obj

    def predict(self, X, **kwargs):
        return self.obj.predict(X, **kwargs)

    def score(self, X, y, scoring=None):
        return self.obj.score(X, y, scoring)
