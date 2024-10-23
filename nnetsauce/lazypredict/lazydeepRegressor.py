import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except ImportError:
    pass
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy
from tqdm import tqdm
import time
try: 
    from sklearn.utils import all_estimators
except ImportError:
    pass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score
from .config import DEEPREGRESSORS
from ..custom import Custom, CustomRegressor

import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

try:
    categorical_transformer_low = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "encoding",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
except TypeError:
    categorical_transformer_low = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "encoding",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # 'OrdinalEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
        ("encoding", OrdinalEncoder()),
    ]
)


# Helper function


def get_card_split(df, cols, n=11):
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


class LazyDeepRegressor(Custom, RegressorMixin):
    """
        Fitting -- almost -- all the regression algorithms with layers of
        nnetsauce's CustomRegressor and returning their scores.

    Parameters:

        verbose: int, optional (default=0)
            Any positive number for verbosity.

        ignore_warnings: bool, optional (default=True)
            When set to True, the warning related to algorigms that are not able to run are ignored.

        custom_metric: function, optional (default=None)
            When function is provided, models are evaluated based on the custom evaluation metric provided.

        predictions: bool, optional (default=False)
            When set to True, the predictions of all the models models are returned as dataframe.

        sort_by: string, optional (default='RMSE')
            Sort models by a metric. Available options are 'R-Squared', 'Adjusted R-Squared', 'RMSE', 'Time Taken' and 'Custom Metric'.
            or a custom metric identified by its name and provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators names or just 'all' (default='all')

        preprocess: bool
            preprocessing is done when set to True

        n_jobs : int, when possible, run in parallel
            For now, only used by individual models that support it.

        n_layers: int, optional (default=3)
            Number of layers of CustomRegressors to be used.

        All the other parameters are the same as CustomRegressor's.
    
    Attributes:

        models_: dict-object
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        
        best_model_: object
            Returns the best model pipeline based on the sort_by metric.                

    Examples:

        import nnetsauce as ns
        import numpy as np
        from sklearn import datasets
        from sklearn.utils import shuffle

        diabetes = datasets.load_diabetes()
        X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
        X = X.astype(np.float32)

        offset = int(X.shape[0] * 0.9)
        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]

        reg = ns.LazyDeepRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
        print(models)

    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        sort_by="RMSE",
        random_state=42,
        estimators="all",
        preprocess=False,
        n_jobs=None,
        # Defining depth
        n_layers=3,
        # CustomRegressor attributes
        obj=None,
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
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.sort_by = sort_by
        self.models_ = {}
        self.best_model_ = None
        self.random_state = random_state
        self.estimators = estimators
        self.preprocess = preprocess
        self.n_layers = n_layers - 1
        self.n_jobs = n_jobs
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

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit Regression algorithms to X_train and y_train, predict and score on X_test, y_test.

        Parameters:

            X_train : array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test : array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            y_train : array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            y_test : array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

        Returns:
        -------
        scores:  Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.

        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.

        """
        R2 = []
        ADJR2 = []
        RMSE = []
        # WIN = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        if self.preprocess is True:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", numeric_transformer, numeric_features),
                    (
                        "categorical_low",
                        categorical_transformer_low,
                        categorical_low,
                    ),
                    (
                        "categorical_high",
                        categorical_transformer_high,
                        categorical_high,
                    ),
                ]
            )
        
        # base models
        try: 
            baseline_names = ["RandomForestRegressor", "XGBRegressor"]
            baseline_models = [RandomForestRegressor(), xgb.XGBRegressor()]
        except Exception as exception:
            baseline_names = ["RandomForestRegressor"]
            baseline_models = [RandomForestRegressor()]

        for name, model in zip(baseline_names, baseline_models):        
            start = time.time()
            try:
                model.fit(X_train, y_train)
                self.models_[name] = model
                y_pred = model.predict(X_test)
                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = adjusted_rsquared(
                    r_squared, X_test.shape[0], X_test.shape[1]
                )
                rmse = np.sqrt(np.mean((y_test - y_pred)**2))

                names.append(name)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)

                if self.custom_metric:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)

                if self.verbose > 0:
                    scores_verbose = {
                        "Model": name,
                        "R-Squared": r_squared,
                        "Adjusted R-Squared": adj_rsquared,
                        "RMSE": rmse,
                        "Time taken": time.time() - start,
                    }

                    if self.custom_metric:
                        scores_verbose[self.custom_metric.__name__] = (
                            custom_metric
                        )

                    print(scores_verbose)
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)                

        if self.estimators == "all":
            self.regressors = DEEPREGRESSORS
        else:
            self.regressors = [
                ("DeepCustomRegressor(" + est[0] + ")", est[1])
                for est in all_estimators()
                if (
                    issubclass(est[1], RegressorMixin)
                    and (est[0] in self.estimators)
                )
            ]

        if self.preprocess is True:

            for name, model in tqdm(self.regressors):  # do parallel exec
                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        layer_regr = CustomRegressor(
                            obj=model(random_state=self.random_state),
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
                    else:
                        layer_regr = CustomRegressor(
                            obj=model(),
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

                    for _ in range(self.n_layers):
                        layer_regr = deepcopy(
                            CustomRegressor(
                                obj=layer_regr,
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

                    layer_regr.fit(X_train, y_train)

                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("regressor", layer_regr),
                        ]
                    )

                    pipe.fit(X_train, y_train)

                    self.models_[name] = pipe
                    y_pred = pipe.predict(X_test)
                    r_squared = r2_score(y_test, y_pred)
                    adj_rsquared = adjusted_rsquared(
                        r_squared, X_test.shape[0], X_test.shape[1]
                    )
                    rmse = np.sqrt(np.mean((y_test - y_pred)**2))

                    names.append(name)
                    R2.append(r_squared)
                    ADJR2.append(adj_rsquared)
                    RMSE.append(rmse)
                    TIME.append(time.time() - start)

                    if self.custom_metric:
                        custom_metric = self.custom_metric(y_test, y_pred)
                        CUSTOM_METRIC.append(custom_metric)

                    if self.verbose > 0:
                        scores_verbose = {
                            "Model": name,
                            "R-Squared": r_squared,
                            "Adjusted R-Squared": adj_rsquared,
                            "RMSE": rmse,
                            "Time taken": time.time() - start,
                        }

                        if self.custom_metric:
                            scores_verbose[self.custom_metric.__name__] = (
                                custom_metric
                            )

                        print(scores_verbose)
                    if self.predictions:
                        predictions[name] = y_pred
                except Exception as exception:
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)

        else:  # no preprocessing

            for name, model in tqdm(self.regressors):  # do parallel exec
                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        layer_regr = CustomRegressor(
                            obj=model(random_state=self.random_state),
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
                    else:
                        layer_regr = CustomRegressor(
                            obj=model(),
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

                    layer_regr.fit(X_train, y_train)

                    for _ in range(self.n_layers):
                        layer_regr = deepcopy(
                            CustomRegressor(
                                obj=layer_regr,
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

                        # layer_regr.fit(X_train, y_train)

                    layer_regr.fit(X_train, y_train)

                    self.models_[name] = layer_regr
                    y_pred = layer_regr.predict(X_test)

                    r_squared = r2_score(y_test, y_pred)
                    adj_rsquared = adjusted_rsquared(
                        r_squared, X_test.shape[0], X_test.shape[1]
                    )
                    rmse = np.sqrt(np.mean((y_test - y_pred)**2))

                    names.append(name)
                    R2.append(r_squared)
                    ADJR2.append(adj_rsquared)
                    RMSE.append(rmse)
                    TIME.append(time.time() - start)

                    if self.custom_metric:
                        custom_metric = self.custom_metric(y_test, y_pred)
                        CUSTOM_METRIC.append(custom_metric)

                    if self.verbose > 0:
                        scores_verbose = {
                            "Model": name,
                            "R-Squared": r_squared,
                            "Adjusted R-Squared": adj_rsquared,
                            "RMSE": rmse,
                            "Time taken": time.time() - start,
                        }

                        if self.custom_metric:
                            scores_verbose[self.custom_metric.__name__] = (
                                custom_metric
                            )

                        print(scores_verbose)
                    if self.predictions:
                        predictions[name] = y_pred
                except Exception as exception:
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)

        scores = {
            "Model": names,
            "Adjusted R-Squared": ADJR2,
            "R-Squared": R2,
            "RMSE": RMSE,
            "Time Taken": TIME,
        }

        if self.custom_metric:
            scores["Custom metric"] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by=self.sort_by, ascending=True).set_index(
            "Model"
        )

        self.best_model_ = self.models_[scores.index[0]]

        if self.predictions is True:
            
            return scores, predictions
        
        return scores

    def get_best_model(self):
        """
        This function returns the best model pipeline based on the sort_by metric.

        Returns:

            best_model: object,
                Returns the best model pipeline based on the sort_by metric.

        """
        return self.best_model_
    
    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.

        Parameters:

            X_train : array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test : array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            y_train : array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            y_test : array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

        Returns:

            models: dict-object,
                Returns a dictionary with each model pipeline as value
                with key as name of models.

        """
        if len(self.models_.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models_


class LazyRegressor(LazyDeepRegressor):
    """
        Fitting -- almost -- all the regression algorithms with 
        nnetsauce's CustomRegressor and returning their scores.

    Parameters:

        verbose: int, optional (default=0)
            Any positive number for verbosity.

        ignore_warnings: bool, optional (default=True)
            When set to True, the warning related to algorigms that are not able to run are ignored.

        custom_metric: function, optional (default=None)
            When function is provided, models are evaluated based on the custom evaluation metric provided.

        predictions: bool, optional (default=False)
            When set to True, the predictions of all the models models are returned as dataframe.

        sort_by: string, optional (default='RMSE')
            Sort models by a metric. Available options are 'R-Squared', 'Adjusted R-Squared', 'RMSE', 'Time Taken' and 'Custom Metric'.
            or a custom metric identified by its name and provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators names or just 'all' (default='all')

        preprocess: bool
            preprocessing is done when set to True

        n_jobs : int, when possible, run in parallel
            For now, only used by individual models that support it.

        All the other parameters are the same as CustomRegressor's.
    
    Attributes:

        models_: dict-object
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        
        best_model_: object
            Returns the best model pipeline based on the sort_by metric.                

    Examples:

        import nnetsauce as ns
        import numpy as np
        from sklearn import datasets
        from sklearn.utils import shuffle

        diabetes = datasets.load_diabetes()
        X, y = shuffle(diabetes.data, diabetes.target, random_state=13)
        X = X.astype(np.float32)

        offset = int(X.shape[0] * 0.9)
        X_train, y_train = X[:offset], y[:offset]
        X_test, y_test = X[offset:], y[offset:]

        reg = ns.LazyRegressor(verbose=0, ignore_warnings=False,
                            custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
        print(models)
    
    """
    
    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        sort_by="RMSE",
        random_state=42,
        estimators="all",
        preprocess=False,
        n_jobs=None,
        # CustomRegressor attributes
        obj=None,
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
            verbose=verbose,
            ignore_warnings=ignore_warnings,
            custom_metric=custom_metric,
            predictions=predictions,
            sort_by=sort_by,
            random_state=random_state,
            estimators=estimators,
            preprocess=preprocess,
            n_jobs=n_jobs,
            n_layers=1,
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