import numpy as np
import pandas as pd
import sklearn.metrics as skm2

try:
    import nnetsauce as ns
except:
    pass
try:
    import GPopt as gp
except:
    pass
from collections import namedtuple
from copy import deepcopy
from tqdm import tqdm
from sklearn import metrics
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import all_estimators
from sklearn.model_selection import cross_val_score
from ..custom import CustomClassifier
from ..predictionset import PredictionSet
from ..utils import matrixops as mo


class DeepClassifier(CustomClassifier, ClassifierMixin):
    """
    Deep Classifier

    Parameters:

        obj: an object
            A base learner, see also https://www.researchgate.net/publication/380701207_Deep_Quasi-Randomized_neural_Networks_for_classification

        n_layers: int (default=3)
            Number of layers. `n_layers = 1` is a simple `CustomClassifier`

        verbose : int, optional (default=0)
            Monitor progress when fitting.

        All the other parameters are nnetsauce `CustomClassifier`'s

    Examples:

        ```python
        import nnetsauce as ns
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegressionCV
        data = load_breast_cancer()
        X = data.data
        y= data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=123)
        obj = LogisticRegressionCV()
        clf = ns.DeepClassifier(obj)
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
        # CustomClassifier attributes
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
        pi_method="icp",
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

        self.stacked_obj = obj
        self.verbose = verbose
        self.n_layers = n_layers

    def fit(self, X, y, sample_weight=None, **kwargs):
        """Fit Classification algorithms to X and y.
        Parameters
        ----------
        X : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        sample_weight: array-like, shape = [n_samples]
                Sample weights.
        Returns
        -------
        A fitted object
        """

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # init layer
        self.stacked_obj = CustomClassifier(
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
                CustomClassifier(
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

        if self.level is not None:
            self.stacked_obj = PredictionSet(
                obj=self.stacked_obj, method=self.pi_method, level=self.level
            )

        try:
            self.stacked_obj.fit(X, y, sample_weight=sample_weight, **kwargs)
        except Exception as e:
            self.stacked_obj.fit(X, y)

        self.obj = deepcopy(self.stacked_obj)

        return self.obj

    def predict(self, X):
        return self.obj.predict(X)

    def predict_proba(self, X):
        return self.obj.predict_proba(X)

    def score(self, X, y, scoring=None):
        return self.obj.score(X, y, scoring)

    def cross_val_optim(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        scoring="accuracy",
        surrogate_obj=None,
        cv=5,
        n_jobs=None,
        n_init=10,
        n_iter=190,
        abs_tol=1e-3,
        verbose=2,
        seed=123,
        **kwargs,
    ):
        """Cross-validation function and hyperparameters' search

        Parameters:

            X_train: array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            y_train: array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test: array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            y_test: array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            scoring: str
                scoring metric; see https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

            surrogate_obj: an object;
                An ML model for estimating the uncertainty around the objective function

            cv: int;
                number of cross-validation folds

            n_jobs: int;
                number of jobs for parallel execution

            n_init: an integer;
                number of points in the initial setting, when `x_init` and `y_init` are not provided

            n_iter: an integer;
                number of iterations of the minimization algorithm

            abs_tol: a float;
                tolerance for convergence of the optimizer (early stopping based on acquisition function)

            verbose: int
                controls verbosity

            seed: int
                reproducibility seed

            **kwargs: dict
                additional parameters to be passed to the estimator

        Examples:

            ```python
            ```
        """

        num_to_activation_name = {1: "relu", 2: "sigmoid", 3: "tanh"}
        num_to_nodes_sim = {1: "sobol", 2: "uniform", 3: "hammersley"}
        num_to_type_clust = {1: "kmeans", 2: "gmm"}

        def deepclassifier_cv(
            X_train,
            y_train,
            # Defining depth
            n_layers=3,
            # CustomClassifier attributes
            n_hidden_features=5,
            activation_name="relu",
            nodes_sim="sobol",
            dropout=0,
            n_clusters=2,
            type_clust="kmeans",
            cv=5,
            n_jobs=None,
            scoring="accuracy",
            seed=123,
        ):
            self.set_params(
                **{
                    "n_layers": n_layers,
                    # CustomClassifier attributes
                    "n_hidden_features": n_hidden_features,
                    "activation_name": activation_name,
                    "nodes_sim": nodes_sim,
                    "dropout": dropout,
                    "n_clusters": n_clusters,
                    "type_clust": type_clust,
                    **kwargs,
                }
            )
            return -cross_val_score(
                estimator=self,
                X=X_train,
                y=y_train,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=0,
            ).mean()

        # objective function for hyperparams tuning
        def crossval_objective(xx):
            return deepclassifier_cv(
                X_train=X_train,
                y_train=y_train,
                # Defining depth
                n_layers=int(np.ceil(xx[0])),
                # CustomClassifier attributes
                n_hidden_features=int(np.ceil(xx[1])),
                activation_name=num_to_activation_name[np.ceil(xx[2])],
                nodes_sim=num_to_nodes_sim[int(np.ceil(xx[3]))],
                dropout=xx[4],
                n_clusters=int(np.ceil(xx[5])),
                type_clust=num_to_type_clust[int(np.ceil(xx[6]))],
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring,
                seed=seed,
            )

        if surrogate_obj is None:
            gp_opt = gp.GPOpt(
                objective_func=crossval_objective,
                lower_bound=np.array([0, 3, 0, 0, 0.0, 0, 0]),
                upper_bound=np.array([5, 100, 3, 3, 0.4, 5, 2]),
                params_names=[
                    "n_layers",
                    # CustomClassifier attributes
                    "n_hidden_features",
                    "activation_name",
                    "nodes_sim",
                    "dropout",
                    "n_clusters",
                    "type_clust",
                ],
                method="bayesian",
                n_init=n_init,
                n_iter=n_iter,
                seed=seed,
            )
        else:
            gp_opt = gp.GPOpt(
                objective_func=crossval_objective,
                lower_bound=np.array([0, 3, 0, 0, 0.0, 0, 0]),
                upper_bound=np.array([5, 100, 3, 3, 0.4, 5, 2]),
                params_names=[
                    "n_layers",
                    # CustomClassifier attributes
                    "n_hidden_features",
                    "activation_name",
                    "nodes_sim",
                    "dropout",
                    "n_clusters",
                    "type_clust",
                ],
                acquisition="ucb",
                method="splitconformal",
                surrogate_obj=ns.PredictionInterval(
                    obj=surrogate_obj, method="splitconformal"
                ),
                n_init=n_init,
                n_iter=n_iter,
                seed=seed,
            )

        res = gp_opt.optimize(verbose=verbose, abs_tol=abs_tol)
        res.best_params["n_layers"] = int(np.ceil(res.best_params["n_layers"]))
        res.best_params["n_hidden_features"] = int(
            np.ceil(res.best_params["n_hidden_features"])
        )
        res.best_params["activation_name"] = num_to_activation_name[
            np.ceil(res.best_params["activation_name"])
        ]
        res.best_params["nodes_sim"] = num_to_nodes_sim[
            int(np.ceil(res.best_params["nodes_sim"]))
        ]
        res.best_params["dropout"] = res.best_params["dropout"]
        res.best_params["n_clusters"] = int(
            np.ceil(res.best_params["n_clusters"])
        )
        res.best_params["type_clust"] = num_to_type_clust[
            int(np.ceil(res.best_params["type_clust"]))
        ]

        # out-of-sample error
        if X_test is not None and y_test is not None:
            self.set_params(**res.best_params, verbose=0, seed=seed)
            preds = self.fit(X_train, y_train).predict(X_test)
            # check error on y_test
            oos_err = getattr(metrics, scoring + "_score")(
                y_true=y_test, y_pred=preds
            )
            result = namedtuple("result", res._fields + ("test_" + scoring,))
            return result(*res, oos_err)
        else:
            return res

    def lazy_cross_val_optim(
        self,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        scoring="accuracy",
        surrogate_objs=None,
        customize=False,
        cv=5,
        n_jobs=None,
        n_init=10,
        n_iter=190,
        abs_tol=1e-3,
        verbose=1,
        seed=123,
    ):
        """Automated Cross-validation function and hyperparameters' search using multiple surrogates

        Parameters:

            X_train: array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            y_train: array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test: array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            y_test: array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            scoring: str
                scoring metric; see https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

            surrogate_objs: object names as a list of strings;
                ML models for estimating the uncertainty around the objective function

            customize: boolean
                if True, the surrogate is transformed into a quasi-randomized network (default is False)

            cv: int;
                number of cross-validation folds

            n_jobs: int;
                number of jobs for parallel execution

            n_init: an integer;
                number of points in the initial setting, when `x_init` and `y_init` are not provided

            n_iter: an integer;
                number of iterations of the minimization algorithm

            abs_tol: a float;
                tolerance for convergence of the optimizer (early stopping based on acquisition function)

            verbose: int
                controls verbosity

            seed: int
                reproducibility seed

        Examples:

            ```python
            ```
        """

        removed_regressors = [
            "TheilSenRegressor",
            "ARDRegression",
            "CCA",
            "GaussianProcessRegressor",
            "GradientBoostingRegressor",
            "HistGradientBoostingRegressor",
            "IsotonicRegression",
            "MultiOutputRegressor",
            "MultiTaskElasticNet",
            "MultiTaskElasticNetCV",
            "MultiTaskLasso",
            "MultiTaskLassoCV",
            "OrthogonalMatchingPursuit",
            "OrthogonalMatchingPursuitCV",
            "PLSCanonical",
            "PLSRegression",
            "RadiusNeighborsRegressor",
            "RegressorChain",
            "StackingRegressor",
            "VotingRegressor",
        ]

        results = []

        for est in all_estimators():

            if surrogate_objs is None:

                if issubclass(est[1], RegressorMixin) and (
                    est[0] not in removed_regressors
                ):
                    try:
                        if customize == True:
                            print(f"\n surrogate: CustomClassifier({est[0]})")
                            surr_obj = ns.CustomClassifier(obj=est[1]())
                        else:
                            print(f"\n surrogate: {est[0]}")
                            surr_obj = est[1]()
                        res = self.cross_val_optim(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            surrogate_obj=surr_obj,
                            cv=cv,
                            n_jobs=n_jobs,
                            scoring=scoring,
                            n_init=n_init,
                            n_iter=n_iter,
                            abs_tol=abs_tol,
                            verbose=verbose,
                            seed=seed,
                        )
                        print(f"\n result: {res}")
                        if customize == True:
                            results.append((f"CustomClassifier({est[0]})", res))
                        else:
                            results.append((est[0], res))
                    except:
                        pass

            else:

                if (
                    issubclass(est[1], RegressorMixin)
                    and (est[0] not in removed_regressors)
                    and est[0] in surrogate_objs
                ):
                    try:
                        if customize == True:
                            print(f"\n surrogate: CustomClassifier({est[0]})")
                            surr_obj = ns.CustomClassifier(obj=est[1]())
                        else:
                            print(f"\n surrogate: {est[0]}")
                            surr_obj = est[1]()
                        res = self.cross_val_optim(
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            surrogate_obj=surr_obj,
                            cv=cv,
                            n_jobs=n_jobs,
                            scoring=scoring,
                            n_init=n_init,
                            n_iter=n_iter,
                            abs_tol=abs_tol,
                            verbose=verbose,
                            seed=seed,
                        )
                        print(f"\n result: {res}")
                        if customize == True:
                            results.append((f"CustomClassifier({est[0]})", res))
                        else:
                            results.append((est[0], res))
                    except:
                        pass

        return results
