# Adapted from: https://github.com/shankarpandala/lazypredict
# Author: Shankar Rao Pandala <shankar.pandala@live.com>

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_pinball_loss,
    mean_absolute_percentage_error
)
from .config import REGRESSORS
from ..mts import MTS
from ..utils import convert_df_to_numeric

import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
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
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
    """
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


class LazyMTS(MTS):
    """
    This module helps in fitting regression models that are available in Scikit-learn to nnetsauce's MTS
    Parameters
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    regressors : list, optional (default="all")
        When function is provided, trains the chosen regressor(s).

    Examples
    --------

    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        regressors="all",
        preprocess=False,
        # MTS attributes
        obj = None,
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
        replications=None,
        kernel=None,
        agg="mean",
        seed=123,
        backend="cpu",
        show_progress=False
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.regressors = regressors
        self.preprocess = preprocess
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
            replications=replications,
            kernel=kernel,
            agg=agg,
            show_progress=show_progress
        )

    def fit(self, X_train, X_test, xreg=None, new_xreg=None, **kwargs):
        """Fit Regression algorithms to X_train, predict and score on X_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
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

        # WIN = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        
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
                    ("categorical_low", categorical_transformer_low, categorical_low),
                    ("categorical_high", categorical_transformer_high, categorical_high),
                ]
            )

        if self.regressors == "all":
            self.regressors = REGRESSORS
        else:
            try:
                temp_list = []
                for regressor in self.regressors:
                    full_name = (regressor.__name__, regressor)
                    temp_list.append(full_name)
                self.regressors = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Regressor(s)")

        if self.preprocess is True:
            for name, model in tqdm(self.regressors): # do parallel exec
                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                ("regressor", MTS(obj=model(random_state=self.random_state, **kwargs),
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
                                            replications=self.replications,
                                            kernel=self.kernel,
                                            agg=self.agg,
                                            seed=self.seed,
                                            backend=self.backend,
                                            show_progress=self.show_progress))])
                    else:
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                ("regressor", MTS(obj=model(**kwargs),
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
                                    replications=self.replications,
                                    kernel=self.kernel,
                                    agg=self.agg,
                                    seed=self.seed,
                                    backend=self.backend,
                                    show_progress=self.show_progress))])

                    pipe.fit(X_train, **kwargs)                                                   
                    #pipe.fit(X_train, xreg=xreg)                

                    self.models[name] = pipe
                    if xreg is not None:
                        assert new_xreg is not None, "xreg and new_xreg must be provided"
                    #X_pred = pipe.predict(h=X_test.shape[0], new_xreg=new_xreg)
                    X_pred = pipe["regressor"].predict(h=X_test.shape[0], **kwargs)
                    rmse = mean_squared_error(X_test, X_pred, squared=False)
                    mae = mean_absolute_error(X_test, X_pred)
                    mpl = mean_pinball_loss(X_test, X_pred)

                    names.append(name)
                    RMSE.append(rmse)
                    MAE.append(mae)
                    MPL.append(mpl)
                    TIME.append(time.time() - start)

                    if self.custom_metric:
                        custom_metric = self.custom_metric(X_test, X_pred)
                        CUSTOM_METRIC.append(custom_metric)

                    if self.verbose > 0:
                        scores_verbose = {
                            "Model": name,
                            #"R-Squared": r_squared,
                            #"Adjusted R-Squared": adj_rsquared,
                            "RMSE": rmse,
                            "MAE": mae,
                            "MPL": mpl,
                            #"MPE": mpe,
                            #"MAPE": mape,
                            "Time taken": time.time() - start,
                        }

                        if self.custom_metric:
                            scores_verbose[self.custom_metric.__name__] = custom_metric

                        print(scores_verbose)
                    if self.predictions:
                        predictions[name] = X_pred
                except Exception as exception:
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)

        else: # no preprocessing

            for name, model in tqdm(self.regressors): # do parallel exec
                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        pipe = MTS(obj=model(random_state=self.random_state, **kwargs),
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
                                            replications=self.replications,
                                            kernel=self.kernel,
                                            agg=self.agg,
                                            seed=self.seed,
                                            backend=self.backend,
                                            show_progress=self.show_progress)                        
                    else:
                        pipe = MTS(obj=model(**kwargs),
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
                                    replications=self.replications,
                                    kernel=self.kernel,
                                    agg=self.agg,
                                    seed=self.seed,
                                    backend=self.backend,
                                    show_progress=self.show_progress)

                    pipe.fit(X_train, **kwargs)                                                   
                    #pipe.fit(X_train, xreg=xreg) # DO xreg like in `ahead`

                    self.models[name] = pipe
                    if xreg is not None:
                        assert new_xreg is not None, "xreg and new_xreg must be provided"

                    if self.preprocess is True: 
                        X_pred = pipe["regressor"].predict(h=X_test.shape[0], **kwargs)
                    else:
                        X_pred = pipe.predict(h=X_test.shape[0], **kwargs) #X_pred = pipe.predict(h=X_test.shape[0], new_xreg=new_xreg) ## DO xreg like in `ahead`
                    
                    rmse = mean_squared_error(X_test, X_pred, squared=False)
                    mae = mean_absolute_error(X_test, X_pred)
                    mpl = mean_pinball_loss(X_test, X_pred)

                    names.append(name)
                    RMSE.append(rmse)
                    MAE.append(mae)
                    MPL.append(mpl)
                    TIME.append(time.time() - start)

                    if self.custom_metric:
                        custom_metric = self.custom_metric(X_test, X_pred)
                        CUSTOM_METRIC.append(custom_metric)

                    if self.verbose > 0:
                        scores_verbose = {
                            "Model": name,
                            #"R-Squared": r_squared,
                            #"Adjusted R-Squared": adj_rsquared,
                            "RMSE": rmse,
                            "MAE": mae,
                            "MPL": mpl,
                            #"MPE": mpe,
                            #"MAPE": mape,
                            "Time taken": time.time() - start,
                        }

                        if self.custom_metric:
                            scores_verbose[self.custom_metric.__name__] = custom_metric

                        print(scores_verbose)
                    if self.predictions:
                        predictions[name] = X_pred
                except Exception as exception:
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)
            
            

        scores = {
            "Model": names,
            #"Adjusted R-Squared": ADJR2,
            #"R-Squared": R2,
            "RMSE": RMSE,
            "MAE": MAE,
            "MPL": MPL,
            #"MPE": MPE,
            #"MAPE": MAPE,
            "Time Taken": TIME,
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by="RMSE", ascending=True).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value 
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test)

        return self.models