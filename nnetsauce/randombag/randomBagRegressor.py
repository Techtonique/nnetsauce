# Authors: Thierry Moudiki
#
# License: BSD 3

import numpy as np
import sklearn.metrics as skm
from .bag import RandomBag
from ..custom import CustomRegressor
from ..utils import misc as mx
from sklearn.base import RegressorMixin
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from .helpers import rbagloop_regression


class RandomBagRegressor(RandomBag, RegressorMixin):
    """Randomized 'Bagging' Regression model

    Parameters:

        obj: object
            any object containing a method fit (obj.fit()) and a method predict
            (obj.predict())

        n_estimators: int
            number of boosting iterations

        n_hidden_features: int
            number of nodes in the hidden layer

        activation_name: str
            activation function: 'relu', 'tanh', 'sigmoid', 'prelu' or 'elu'

        a: float
            hyperparameter for 'prelu' or 'elu' activation function

        nodes_sim: str
            type of simulation for the nodes: 'sobol', 'hammersley', 'halton',
            'uniform'

        bias: boolean
            indicates if the hidden layer contains a bias term (True) or not
            (False)

        dropout: float
            regularization parameter; (random) percentage of nodes dropped out
            of the training

        direct_link: boolean
            indicates if the original predictors are included (True) in model''s
            fitting or not (False)

        n_clusters: int
            number of clusters for 'kmeans' or 'gmm' clustering (could be 0:
                no clustering)

        cluster_encode: bool
            defines how the variable containing clusters is treated (default is one-hot)
            if `False`, then labels are used, without one-hot encoding

        type_clust: str
            type of clustering method: currently k-means ('kmeans') or Gaussian
            Mixture Model ('gmm')

        type_scaling: a tuple of 3 strings
            scaling methods for inputs, hidden layer, and clustering respectively
            (and when relevant).
            Currently available: standardization ('std') or MinMax scaling ('minmax')

        col_sample: float
            percentage of covariates randomly chosen for training

        row_sample: float
            percentage of rows chosen for training, by stratified bootstrapping

        seed: int
            reproducibility seed for nodes_sim=='uniform'

        backend: str
            "cpu" or "gpu" or "tpu"

    Attributes:

        voter_: dict
            dictionary containing all the fitted base-learners


    Examples:

    ```python
    import numpy as np
    import nnetsauce as ns
    from sklearn.datasets import fetch_california_housing
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split

    X, y = fetch_california_housing(return_X_y=True, as_frame=False)

    # split data into training test and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, random_state=13)

    # Requires further tuning
    obj = DecisionTreeRegressor(max_depth=3, random_state=123)
    obj2 = ns.RandomBagRegressor(obj=obj, direct_link=False,
                                n_estimators=50,
                                col_sample=0.9, row_sample=0.9,
                                dropout=0, n_clusters=0, verbose=1)

    obj2.fit(X_train, y_train)

    print(np.sqrt(obj2.score(X_test, y_test))) # RMSE

    ```

    """

    # construct the object -----

    def __init__(
        self,
        obj,
        n_estimators=10,
        n_hidden_features=1,
        activation_name="relu",
        a=0.01,
        nodes_sim="sobol",
        bias=True,
        dropout=0,
        direct_link=False,
        n_clusters=2,
        cluster_encode=True,
        type_clust="kmeans",
        type_scaling=("std", "std", "std"),
        col_sample=1,
        row_sample=1,
        n_jobs=None,
        seed=123,
        verbose=1,
        backend="cpu",
    ):
        super().__init__(
            obj=obj,
            n_estimators=n_estimators,
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

        self.type_fit = "regression"
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.voter_ = {}

    def fit(self, X, y, **kwargs):
        """Fit Random 'Bagging' model to training data (X, y).

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            y: array-like, shape = [n_samples]
                Target values.

            **kwargs: additional parameters to be passed to
                    self.cook_training_set or self.obj.fit

        Returns:

            self: object

        """

        base_learner = CustomRegressor(
            self.obj,
            n_hidden_features=self.n_hidden_features,
            activation_name=self.activation_name,
            a=self.a,
            nodes_sim=self.nodes_sim,
            bias=self.bias,
            dropout=self.dropout,
            direct_link=self.direct_link,
            n_clusters=self.n_clusters,
            type_clust=self.type_clust,
            type_scaling=self.type_scaling,
            col_sample=self.col_sample,
            row_sample=self.row_sample,
            seed=self.seed,
        )

        # 1 - Sequential training -----

        if self.n_jobs is None:
            self.voter_ = rbagloop_regression(
                base_learner, X, y, self.n_estimators, self.verbose, self.seed
            )

            self.n_estimators = len(self.voter_)

            return self

        # 2 - Parallel training -----
        # buggy
        # if self.n_jobs is not None:
        def fit_estimators(m):
            base_learner__ = pickle.loads(pickle.dumps(base_learner, -1))
            base_learner__.set_params(seed=self.seed + m * 1000)
            base_learner__.fit(X, y, **kwargs)
            return base_learner__

        if self.verbose == 1:
            voters_list = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(fit_estimators)(m)
                for m in tqdm(range(self.n_estimators))
            )
        else:
            voters_list = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(fit_estimators)(m) for m in range(self.n_estimators)
            )

        self.voter_ = {i: elt for i, elt in enumerate(voters_list)}

        self.n_estimators = len(self.voter_)

        return self

    def predict(self, X, weights=None, **kwargs):
        """Predict for test data X.

        Args:

            X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number
                of samples and n_features is the number of features.

            **kwargs: additional parameters to be passed to
                    self.cook_test_set

        Returns:

            estimates for test data: {array-like}

        """

        def calculate_preds(voter, weights=None):
            ensemble_preds = 0

            n_iter = len(voter)

            assert n_iter > 0, "no estimator found in `RandomBag` ensemble"

            if weights is None:
                for idx, elt in voter.items():
                    ensemble_preds += elt.predict(X)

                return ensemble_preds / n_iter

            # if weights is not None:
            for idx, elt in voter.items():
                ensemble_preds += weights[idx] * elt.predict(X)

            return ensemble_preds

        # end calculate_preds ----

        if weights is None:
            return calculate_preds(self.voter_)

        # if weights is not None:
        self.weights = weights

        return calculate_preds(self.voter_, weights)
