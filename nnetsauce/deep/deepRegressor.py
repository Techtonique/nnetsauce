import numpy as np
import pandas as pd
import sklearn.metrics as skm2
from copy import deepcopy
from tqdm import tqdm
from sklearn.base import RegressorMixin
from ..custom import CustomRegressor
from ..predictioninterval import PredictionInterval
from ..utils import matrixops as mo


class DeepRegressor(CustomRegressor, RegressorMixin):
    """
    Deep Regressor

    Parameters:

        obj: an object
            A base learner, see also https://www.researchgate.net/publication/380701207_Deep_Quasi-Randomized_neural_Networks_for_classification

        verbose : int, optional (default=0)
            Monitor progress when fitting.

        n_layers: int (default=3)
            Number of layers. `n_layers = 1` is a simple `CustomRegressor`

        All the other parameters are nnetsauce `CustomRegressor`'s

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
        # Defining depth
        n_layers=3,
        verbose=0,
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
        level=None,
        pi_method="splitconformal",
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
            level=level,
            pi_method=pi_method,
            seed=seed,
            backend=backend,
        )

        assert n_layers >= 1, "must have n_layers >= 1"

        self.stacked_obj = deepcopy(obj)
        self.verbose = verbose
        self.n_layers = n_layers
        self.level = level
        self.pi_method = pi_method
        self.coef_ = None

    def fit(self, X, y, **kwargs):
        """Fit Regression algorithms to X and y.
        Parameters
        ----------
        X : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        **kwargs: dict
            Additional parameters to be passed to the fit method
            of the base learner. For example, `sample_weight`.
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
            self.stacked_obj.fit(X, y, **kwargs)

        if self.level is not None:
            self.stacked_obj = PredictionInterval(
                obj=self.stacked_obj, method=self.pi_method, level=self.level
            )

        if hasattr(self.stacked_obj, "clustering_obj_"):
            self.clustering_obj_ = self.stacked_obj.clustering_obj_

        if hasattr(self.stacked_obj, "coef_"):
            self.coef_ = self.stacked_obj.coef_

        if hasattr(self.stacked_obj, "scaler_"):
            self.scaler_ = self.stacked_obj.scaler_

        if hasattr(self.stacked_obj, "nn_scaler_"):
            self.nn_scaler_ = self.stacked_obj.nn_scaler_

        if hasattr(self.stacked_obj, "clustering_scaler_"):
            self.clustering_scaler_ = self.stacked_obj.clustering_scaler_

        return self

    def partial_fit(self, X, y, **kwargs):
        """Fit Regression algorithms to X and y.
        Parameters
        ----------
        X : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        **kwargs: dict
            Additional parameters to be passed to the fit method
            of the base learner. For example, `sample_weight`.
        Returns
        -------
        A fitted object
        """
        assert hasattr(self, "stacked_obj"), "model must be fitted first"
        current_obj = self.stacked_obj
        for _ in range(self.n_layers):
            try:
                input_X = current_obj.obj.cook_test_set(X)
                current_obj.obj.partial_fit(input_X, y, **kwargs)
                try:
                    current_obj = current_obj.obj
                except AttributeError:
                    pass
            except ValueError as e:
                print(e)
                pass
        return self

    def predict(self, X, **kwargs):
        if self.level is not None:
            return self.stacked_obj.predict(X, return_pi=True)
        return self.stacked_obj.predict(X, **kwargs)

    def score(self, X, y, scoring=None):
        return self.stacked_obj.score(X, y, scoring)
