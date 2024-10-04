import numpy as np
import pandas as pd
import time

from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import RegressorMixin
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_pinball_loss,
    mean_absolute_percentage_error,
)
from tqdm import tqdm

from .config import DEEPREGRESSORSMTS, REGRESSORSMTS
from ..deep import DeepMTS
from ..mts import ClassicalMTS, MTS
from ..utils import (
    convert_df_to_numeric,
    coverage,
    dict_to_dataframe_series,
    mean_errors,
    winkler_score,
)

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


# def mean_percentage_error(y_pred, y_true):
#     pe = (y_pred - y_true) / np.maximum(np.abs(y_true),
#                                         np.finfo(np.float64).eps)
#     return np.average(pe)


class LazyDeepMTS(MTS):
    """

    Fitting -- almost -- all the regression algorithms with layers of
    nnetsauce's CustomRegressor to multivariate time series
    and returning their scores.

    Parameters:

        verbose: int, optional (default=0)
            Any positive number for verbosity.

        ignore_warnings: bool, optional (default=True)
            When set to True, the warning related to algorigms that are not
            able to run are ignored.

        custom_metric: function, optional (default=None)
            When function is provided, models are evaluated based on the custom
              evaluation metric provided.

        predictions: bool, optional (default=False)
            When set to True, the predictions of all the models models are returned as dataframe.

        sort_by: string, optional (default='RMSE')
            Sort models by a metric. Available options are 'RMSE', 'MAE', 'MPL', 'MPE', 'MAPE',
            'R-Squared', 'Adjusted R-Squared' or a custom metric identified by its name and
            provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators (regression algorithms) names or just 'all' (default='all')

        preprocess: bool, preprocessing is done when set to True

        n_layers: int, optional (default=1)
            Number of layers in the network. When set to 1, the model is equivalent to a MTS.

        h: int, optional (default=None)
            Number of steps ahead to predict (when used, must be > 0 and < X_test.shape[0]).

        All the other parameters are the same as MTS's.
    
    Attributes:

        models_: dict-object
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        
        best_model_: object
            Returns the best model pipeline based on the sort_by metric.

    Examples:

        See https://thierrymoudiki.github.io/blog/2023/10/29/python/quasirandomizednn/MTS-LazyPredict

    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        sort_by=None, # leave it as is 
        random_state=42,
        estimators="all",
        preprocess=False,
        n_layers=1,
        h=None,
        # MTS attributes
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
        lags=15,
        type_pi="scp2-kde",
        block_size=None,
        replications=None,
        kernel=None,
        agg="mean",
        seed=123,
        backend="cpu",
        show_progress=False,
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
        self.n_layers = n_layers
        self.h = h
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
            seed=seed,
            backend=backend,
            lags=lags,
            type_pi=type_pi,
            block_size=block_size,
            replications=replications,
            kernel=kernel,
            agg=agg,
            verbose=verbose,
            show_progress=show_progress,
        )
        if self.replications is not None or self.type_pi == "gaussian":
            if self.sort_by is None:
                self.sort_by = "WINKLERSCORE"
        else:
            if self.sort_by is None:
                self.sort_by = "RMSE"

    def fit(self, X_train, X_test, xreg=None, per_series=False, **kwargs):
        """Fit Regression algorithms to X_train, predict and score on X_test.

        Parameters:

            X_train: array-like or data frame,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test: array-like or data frame,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            xreg: array-like, optional (default=None)
                Additional (external) regressors to be passed to self.obj
                xreg must be in 'increasing' order (most recent observations last)

            per_series: bool, optional (default=False)
                When set to True, the metrics are computed series by series.

            **kwargs: dict, optional (default=None)
                Additional parameters to be passed to `fit` method of `obj`.

        Returns:

            scores: Pandas DataFrame
                Returns metrics of all the models in a Pandas DataFrame.

            predictions: Pandas DataFrame
                Returns predictions of all the models in a Pandas DataFrame.

        """
        R2 = []
        ADJR2 = []
        ME = []
        MPL = []
        RMSE = []
        MAE = []
        MPE = []
        MAPE = []
        WINKLERSCORE = []
        COVERAGE = []

        # WIN = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric is not None:
            CUSTOM_METRIC = []

        if self.h is None:
            assert X_test is not None, "If h is None, X_test must be provided."

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        self.series_names = X_train.columns.tolist()

        X_train = convert_df_to_numeric(X_train)
        X_test = convert_df_to_numeric(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        if self.preprocess:
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

        # baselines (Classical MTS) ----
        for i, name in enumerate(["ARIMA", "ETS", "Theta", "VAR", "VECM"]):           
            try: 
                start = time.time()
                regr = ClassicalMTS(model=name)
                regr.fit(X_train, **kwargs)
                self.models_[name] = regr
                if self.h is None:
                    X_pred = regr.predict(h=X_test.shape[0], **kwargs)
                else:
                    assert self.h > 0, "h must be > 0"
                    X_pred = regr.predict(h=self.h, **kwargs)
                    try:
                        X_test = X_test[0: self.h, :]
                    except Exception as e:
                        X_test = X_test.iloc[0: self.h, :]

                if per_series == False:
                    rmse = mean_squared_error(X_test, X_pred.mean, squared=False)
                    mae = mean_absolute_error(X_test, X_pred.mean)
                    mpl = mean_pinball_loss(X_test, X_pred.mean)
                else:
                    rmse = mean_errors(
                        actual=X_test,
                        pred=X_pred,
                        scoring="root_mean_squared_error",
                        per_series=True,
                    )
                    mae = mean_errors(
                        actual=X_test,
                        pred=X_pred,
                        scoring="mean_absolute_error",
                        per_series=True,
                    )
                    mpl = mean_errors(
                        actual=X_test,
                        pred=X_pred,
                        scoring="mean_pinball_loss",
                        per_series=True,
                    )
            except Exception as exception:
                continue

            names.append(name)
            RMSE.append(rmse)
            MAE.append(mae)
            MPL.append(mpl)

            if self.custom_metric is not None:                        
                try: 
                    if self.h is None:
                        custom_metric = self.custom_metric(X_test, X_pred)
                    else:
                        custom_metric = self.custom_metric(X_test_h, X_pred)
                    CUSTOM_METRIC.append(custom_metric)
                except Exception as e:
                    custom_metric = np.iinfo(np.float32).max
                    CUSTOM_METRIC.append(np.iinfo(np.float32).max)

            if (self.replications is not None) or (self.type_pi == "gaussian"):
                if per_series == False:
                    winklerscore = winkler_score(
                        obj=X_pred, actual=X_test, level=95
                    )
                    coveragecalc = coverage(X_pred, X_test, level=95)
                else: 
                    winklerscore = winkler_score(
                        obj=X_pred, actual=X_test, level=95, per_series=True
                    )
                    coveragecalc = coverage(X_pred, X_test, level=95, per_series=True)
                WINKLERSCORE.append(winklerscore)
                COVERAGE.append(coveragecalc)
            TIME.append(time.time() - start)        
        
        if self.estimators == "all":
            if self.n_layers <= 1:
                self.regressors = REGRESSORSMTS
            else:
                self.regressors = DEEPREGRESSORSMTS
        else:
            if self.n_layers <= 1:
                self.regressors = [
                    ("MTS(" + est[0] + ")", est[1])
                    for est in all_estimators()
                    if (
                        issubclass(est[1], RegressorMixin)
                        and (est[0] in self.estimators)
                    )
                ]
            else:  # self.n_layers > 1
                self.regressors = [
                    ("DeepMTS(" + est[0] + ")", est[1])
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
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                (
                                    "regressor",
                                    DeepMTS(
                                        obj=model(
                                            random_state=self.random_state,
                                            **kwargs
                                        ),
                                        n_layers=self.n_layers,
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
                                        lags=self.lags,
                                        type_pi=self.type_pi,
                                        block_size=self.block_size,
                                        replications=self.replications,
                                        kernel=self.kernel,
                                        agg=self.agg,
                                        seed=self.seed,
                                        backend=self.backend,
                                        show_progress=self.show_progress,
                                    ),
                                ),
                            ]
                        )
                    else:  # "random_state" in model().get_params().keys()
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                (
                                    "regressor",
                                    DeepMTS(
                                        obj=model(**kwargs),
                                        n_layers=self.n_layers,
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
                                        lags=self.lags,
                                        type_pi=self.type_pi,
                                        block_size=self.block_size,
                                        replications=self.replications,
                                        kernel=self.kernel,
                                        agg=self.agg,
                                        seed=self.seed,
                                        backend=self.backend,
                                        show_progress=self.show_progress,
                                    ),
                                ),
                            ]
                        )

                    pipe.fit(X_train, **kwargs)
                    # pipe.fit(X_train, xreg=xreg)

                    self.models_[name] = pipe

                    if self.h is None:
                        X_pred = pipe["regressor"].predict(h=self.h, **kwargs)
                    else:
                        assert self.h > 0, "h must be > 0"
                        X_pred = pipe["regressor"].predict(h=self.h, **kwargs)

                    if (self.replications is not None) or (
                        self.type_pi == "gaussian"
                    ):
                        if per_series == False:
                            rmse = mean_squared_error(
                                X_test, X_pred.mean, squared=False
                            )
                            mae = mean_absolute_error(X_test, X_pred.mean)
                            mpl = mean_pinball_loss(X_test, X_pred.mean)
                            winklerscore = winkler_score(
                                obj=X_pred, actual=X_test, level=95
                            )
                            coveragecalc = coverage(X_pred, X_test, level=95)
                        else:
                            rmse = mean_errors(
                                actual=X_test,
                                pred=X_pred,
                                scoring="root_mean_squared_error",
                                per_series=True,
                            )
                            mae = mean_errors(
                                actual=X_test,
                                pred=X_pred,
                                scoring="mean_absolute_error",
                                per_series=True,
                            )
                            mpl = mean_errors(
                                actual=X_test,
                                pred=X_pred,
                                scoring="mean_pinball_loss",
                                per_series=True,
                            )
                            winklerscore = winkler_score(
                                obj=X_pred,
                                actual=X_test,
                                level=95,
                                per_series=True,
                            )
                            coveragecalc = coverage(
                                X_pred, X_test, level=95, per_series=True
                            )
                    else:
                        if per_series == False:
                            rmse = mean_squared_error(
                                X_test, X_pred, squared=False
                            )
                            mae = mean_absolute_error(X_test, X_pred)
                            mpl = mean_pinball_loss(X_test, X_pred)
                        else:
                            rmse = mean_errors(
                                actual=X_test,
                                pred=X_pred,
                                scoring="root_mean_squared_error",
                                per_series=True,
                            )
                            mae = mean_errors(
                                actual=X_test,
                                pred=X_pred,
                                scoring="mean_absolute_error",
                                per_series=True,
                            )
                            mpl = mean_errors(
                                actual=X_test,
                                pred=X_pred,
                                scoring="mean_pinball_loss",
                                per_series=True,
                            )

                    names.append(name)
                    RMSE.append(rmse)
                    MAE.append(mae)
                    MPL.append(mpl)

                    if (self.replications is not None) or (
                        self.type_pi == "gaussian"
                    ):
                        WINKLERSCORE.append(winklerscore)
                        COVERAGE.append(coveragecalc)
                    TIME.append(time.time() - start)

                    if self.custom_metric is not None:                        
                        try: 
                            custom_metric = self.custom_metric(X_test, X_pred)
                            CUSTOM_METRIC.append(custom_metric)
                        except Exception as e:
                            custom_metric = np.iinfo(np.float32).max
                            CUSTOM_METRIC.append(custom_metric)

                    if self.verbose > 0:
                        if (self.replications is not None) or (
                            self.type_pi == "gaussian"
                        ):
                            scores_verbose = {
                                "Model": name,
                                "RMSE": rmse,
                                "MAE": mae,
                                "MPL": mpl,
                                "WINKLERSCORE": winklerscore,
                                "COVERAGE": coveragecalc,
                                "Time taken": time.time() - start,
                            }
                        else:
                            scores_verbose = {
                                "Model": name,
                                "RMSE": rmse,
                                "MAE": mae,
                                "MPL": mpl,
                                "Time taken": time.time() - start,
                            }

                        if self.custom_metric is not None:
                            scores_verbose["Custom metric"] = custom_metric
                            
                    if self.predictions:
                        predictions[name] = X_pred
                except Exception as exception:
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)

        else:  # no preprocessing

            for name, model in tqdm(self.regressors):  # do parallel exec
                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        pipe = DeepMTS(
                            obj=model(random_state=self.random_state, **kwargs),
                            n_layers=self.n_layers,
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
                            lags=self.lags,
                            type_pi=self.type_pi,
                            block_size=self.block_size,
                            replications=self.replications,
                            kernel=self.kernel,
                            agg=self.agg,
                            seed=self.seed,
                            backend=self.backend,
                            show_progress=self.show_progress,
                        )
                    else:
                        pipe = DeepMTS(
                            obj=model(**kwargs),
                            n_layers=self.n_layers,
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
                            lags=self.lags,
                            type_pi=self.type_pi,
                            block_size=self.block_size,
                            replications=self.replications,
                            kernel=self.kernel,
                            agg=self.agg,
                            seed=self.seed,
                            backend=self.backend,
                            show_progress=self.show_progress,
                        )

                    pipe.fit(X_train, xreg, **kwargs)
                    # pipe.fit(X_train, xreg=xreg) # DO xreg like in `ahead`

                    self.models_[name] = pipe

                    if self.preprocess is True:
                        if self.h is None:
                            X_pred = pipe["regressor"].predict(
                                h=X_test.shape[0], **kwargs
                            )
                        else:
                            assert (
                                self.h > 0 and self.h <= X_test.shape[0]
                            ), "h must be > 0 and < X_test.shape[0]"
                            X_pred = pipe["regressor"].predict(
                                h=self.h, **kwargs
                            )

                    else:

                        if self.h is None:
                            X_pred = pipe.predict(
                                h=X_test.shape[0], **kwargs
                            )  # X_pred = pipe.predict(h=X_test.shape[0], new_xreg=new_xreg) ## DO xreg like in `ahead`
                        else:
                            assert (
                                self.h > 0 and self.h <= X_test.shape[0]
                            ), "h must be > 0 and < X_test.shape[0]"
                            X_pred = pipe.predict(h=self.h, **kwargs)

                    if self.h is None:
                        if (self.replications is not None) or (
                            self.type_pi == "gaussian"
                        ):

                            if per_series == True:
                                rmse = mean_errors(
                                    actual=X_test,
                                    pred=X_pred.mean,
                                    scoring="root_mean_squared_error",
                                    per_series=True,
                                )
                                mae = mean_errors(
                                    actual=X_test,
                                    pred=X_pred.mean,
                                    scoring="mean_absolute_error",
                                    per_series=True,
                                )
                                mpl = mean_errors(
                                    actual=X_test,
                                    pred=X_pred.mean,
                                    scoring="mean_pinball_loss",
                                    per_series=True,
                                )
                                winklerscore = winkler_score(
                                    obj=X_pred,
                                    actual=X_test,
                                    level=95,
                                    per_series=True,
                                )
                                coveragecalc = coverage(
                                    X_pred, X_test, level=95, per_series=True
                                )
                            else:
                                rmse = mean_squared_error(
                                    X_test, X_pred.mean, squared=False
                                )
                                mae = mean_absolute_error(X_test, X_pred.mean)
                                mpl = mean_pinball_loss(X_test, X_pred.mean)
                                winklerscore = winkler_score(
                                    obj=X_pred, actual=X_test, level=95
                                )
                                coveragecalc = coverage(
                                    X_pred, X_test, level=95
                                )
                        else:  # no prediction interval
                            if per_series == True:
                                rmse = mean_errors(
                                    actual=X_test,
                                    pred=X_pred,
                                    scoring="root_mean_squared_error",
                                    per_series=True,
                                )
                                mae = mean_errors(
                                    actual=X_test,
                                    pred=X_pred,
                                    scoring="mean_absolute_error",
                                    per_series=True,
                                )
                                mpl = mean_errors(
                                    actual=X_test,
                                    pred=X_pred,
                                    scoring="mean_pinball_loss",
                                    per_series=True,
                                )
                            else:
                                rmse = mean_squared_error(
                                    X_test, X_pred, squared=False
                                )
                                mae = mean_absolute_error(X_test, X_pred)
                                mpl = mean_pinball_loss(X_test, X_pred)
                    else:  # self.h is not None
                        if (self.replications is not None) or (
                            self.type_pi == "gaussian"
                        ):

                            if per_series == False:
                                if isinstance(X_test, pd.DataFrame) == False:
                                    X_test_h = X_test[0: self.h, :]
                                    rmse = mean_squared_error(
                                        X_test_h, X_pred.mean, squared=False
                                    )
                                    mae = mean_absolute_error(
                                        X_test_h, X_pred.mean
                                    )
                                    mpl = mean_pinball_loss(
                                        X_test_h, X_pred.mean
                                    )
                                    winklerscore = winkler_score(
                                        obj=X_pred, actual=X_test_h, level=95
                                    )
                                    coveragecalc = coverage(
                                        X_pred, X_test_h, level=95
                                    )
                                else:
                                    X_test_h = X_test.iloc[0: self.h, :]
                                    rmse = mean_squared_error(
                                        X_test_h, X_pred.mean, squared=False
                                    )
                                    mae = mean_absolute_error(
                                        X_test_h, X_pred.mean
                                    )
                                    mpl = mean_pinball_loss(
                                        X_test_h, X_pred.mean
                                    )
                                    winklerscore = winkler_score(
                                        obj=X_pred, actual=X_test_h, level=95
                                    )
                                    coveragecalc = coverage(
                                        X_pred, X_test_h, level=95
                                    )
                            else:
                                if isinstance(X_test, pd.DataFrame):
                                    X_test_h = X_test.iloc[0: self.h, :]
                                    rmse = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="root_mean_squared_error",
                                        per_series=True,
                                    )
                                    mae = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="mean_absolute_error",
                                        per_series=True,
                                    )
                                    mpl = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="mean_pinball_loss",
                                        per_series=True,
                                    )
                                    winklerscore = winkler_score(
                                        obj=X_pred,
                                        actual=X_test_h,
                                        level=95,
                                        per_series=True,
                                    )
                                    coveragecalc = coverage(
                                        X_pred,
                                        X_test_h,
                                        level=95,
                                        per_series=True,
                                    )
                                else:
                                    X_test_h = X_test[0: self.h, :]
                                    rmse = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="root_mean_squared_error",
                                        per_series=True,
                                    )
                                    mae = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="mean_absolute_error",
                                        per_series=True,
                                    )
                                    mpl = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="mean_pinball_loss",
                                        per_series=True,
                                    )
                                    winklerscore = winkler_score(
                                        obj=X_pred,
                                        actual=X_test_h,
                                        level=95,
                                        per_series=True,
                                    )
                                    coveragecalc = coverage(
                                        X_pred,
                                        X_test_h,
                                        level=95,
                                        per_series=True,
                                    )
                        else:  # no prediction interval

                            if per_series == False:
                                if isinstance(X_test, pd.DataFrame):
                                    X_test_h = X_test.iloc[0: self.h, :]
                                    rmse = mean_squared_error(
                                        X_test_h, X_pred, squared=False
                                    )
                                    mae = mean_absolute_error(X_test_h, X_pred)
                                    mpl = mean_pinball_loss(X_test_h, X_pred)
                                else:
                                    X_test_h = X_test[0: self.h, :]
                                    rmse = mean_squared_error(
                                        X_test_h, X_pred, squared=False
                                    )
                                    mae = mean_absolute_error(X_test_h, X_pred)
                                    mpl = mean_pinball_loss(X_test_h, X_pred)
                            else:
                                if isinstance(X_test, pd.DataFrame):
                                    X_test_h = X_test.iloc[0: self.h, :]
                                    rmse = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="root_mean_squared_error",
                                        per_series=True,
                                    )
                                    mae = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="mean_absolute_error",
                                        per_series=True,
                                    )
                                    mpl = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="mean_pinball_loss",
                                        per_series=True,
                                    )
                                else:
                                    X_test_h = X_test[0: self.h, :]
                                    rmse = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="root_mean_squared_error",
                                        per_series=True,
                                    )
                                    mae = mean_errors(
                                        actual=X_test_h,
                                        pred=X_pred,
                                        scoring="mean_absolute_error",
                                        per_series=True,
                                    )

                    names.append(name)
                    RMSE.append(rmse)
                    MAE.append(mae)
                    MPL.append(mpl)
                    if (self.replications is not None) or (
                        self.type_pi == "gaussian"
                    ):
                        WINKLERSCORE.append(winklerscore)
                        COVERAGE.append(coveragecalc)
                    TIME.append(time.time() - start)

                    if self.custom_metric is not None:                        
                        try: 
                            if self.h is None:
                                custom_metric = self.custom_metric(X_test, X_pred)
                            else:
                                custom_metric = self.custom_metric(X_test_h, X_pred)
                            CUSTOM_METRIC.append(custom_metric)
                        except Exception as e:
                            custom_metric = np.iinfo(np.float32).max
                            CUSTOM_METRIC.append(np.iinfo(np.float32).max)

                    if self.verbose > 0:
                        if (self.replications is not None) or (
                            self.type_pi == "gaussian"
                        ):
                            scores_verbose = {
                                "Model": name,
                                "RMSE": rmse,
                                "MAE": mae,
                                "MPL": mpl,
                                "WINKLERSCORE": winklerscore,
                                "COVERAGE": coveragecalc,
                                "Time taken": time.time() - start,
                            }
                        else:
                            scores_verbose = {
                                "Model": name,
                                "RMSE": rmse,
                                "MAE": mae,
                                "MPL": mpl,
                                "Time taken": time.time() - start,
                            }

                        if self.custom_metric is not None:
                            scores_verbose["Custom metric"] = custom_metric                            
                       
                    if self.predictions:
                        predictions[name] = X_pred

                except Exception as exception:
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)

        if (self.replications is not None) or (self.type_pi == "gaussian"):
            scores = {
                "Model": names,
                "RMSE": RMSE,
                "MAE": MAE,
                "MPL": MPL,
                "WINKLERSCORE": WINKLERSCORE,
                "COVERAGE": COVERAGE,
                "Time Taken": TIME,
            }
        else:
            scores = {
                "Model": names,
                "RMSE": RMSE,
                "MAE": MAE,
                "MPL": MPL,
                "Time Taken": TIME,
            }

        if self.custom_metric is not None:
            scores["Custom metric"] = CUSTOM_METRIC
        
        if per_series:
            scores = dict_to_dataframe_series(scores, self.series_names)
        else:
            scores = pd.DataFrame(scores)
       
        try: # case per_series, can't be sorted
            scores = scores.sort_values(by=self.sort_by, ascending=True).set_index("Model")
            
            self.best_model_ = self.models_[scores.index[0]]
        except Exception as e:
            pass 

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)

        return scores, predictions_df if self.predictions is True else scores

    def get_best_model(self):
        """
        This function returns the best model pipeline based on the sort_by metric.

        Returns:

            best_model: object,
                Returns the best model pipeline based on the sort_by metric.

        """
        return self.best_model_

    def provide_models(self, X_train, X_test):
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

        Returns:

            models: dict-object,
                Returns a dictionary with each model pipeline as value
                with key as name of models.

        """
        if self.h is None:
            if len(self.models_.keys()) == 0:
                self.fit(X_train, X_test)
        else:
            if len(self.models_.keys()) == 0:
                if isinstance(X_test, pd.DataFrame):
                    self.fit(X_train, X_test.iloc[0: self.h, :])
                else:
                    self.fit(X_train, X_test[0: self.h, :])

        return self.models_

class LazyMTS(LazyDeepMTS):
    """
    Fitting -- almost -- all the regression algorithms to multivariate time series
    and returning their scores (no layers).

    Parameters:

        verbose: int, optional (default=0)
            Any positive number for verbosity.

        ignore_warnings: bool, optional (default=True)
            When set to True, the warning related to algorigms that are not
            able to run are ignored.

        custom_metric: function, optional (default=None)
            When function is provided, models are evaluated based on the custom
              evaluation metric provided.

        predictions: bool, optional (default=False)
            When set to True, the predictions of all the models models are returned as dataframe.

        sort_by: string, optional (default='RMSE')
            Sort models by a metric. Available options are 'RMSE', 'MAE', 'MPL', 'MPE', 'MAPE',
            'R-Squared', 'Adjusted R-Squared' or a custom metric identified by its name and
            provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators (regression algorithms) names or just 'all' (default='all')

        preprocess: bool, preprocessing is done when set to True

        h: int, optional (default=None)
            Number of steps ahead to predict (when used, must be > 0 and < X_test.shape[0]).

        All the other parameters are the same as MTS's.
    
    Attributes:

        models_: dict-object
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        
        best_model_: object
            Returns the best model pipeline based on the sort_by metric.

    Examples:

        See https://thierrymoudiki.github.io/blog/2023/10/29/python/quasirandomizednn/MTS-LazyPredict

    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        sort_by=None, # leave it as is 
        random_state=42,
        estimators="all",
        preprocess=False,
        h=None,
        # MTS attributes
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
        lags=15,
        type_pi="scp2-kde",
        block_size=None,
        replications=None,
        kernel=None,
        agg="mean",
        seed=123,
        backend="cpu",
        show_progress=False,
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
            n_layers=1,
            h=h,
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
            lags=lags,
            type_pi=type_pi,
            block_size=block_size,
            replications=replications,
            kernel=kernel,
            agg=agg,
            seed=seed,
            backend=backend,
            show_progress=show_progress,
        )
