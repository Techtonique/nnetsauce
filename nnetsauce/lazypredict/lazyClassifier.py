import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
import time
from ..multitask import MultitaskClassifier, SimpleMultitaskClassifier
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)
from .config import (
    CLASSIFIERS,
    MULTITASKCLASSIFIERS,
    SIMPLEMULTITASKCLASSIFIERS,
)
from ..custom import Custom, CustomClassifier
from ..utils.misc import flatten

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


def get_card_split(df, cols, n=11):
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


class LazyClassifier(Custom, ClassifierMixin):
    """

        Fitting -- almost -- all the classification algorithms with nnetsauce's
        CustomClassifier and returning their scores.

    Parameters:

        verbose: int, optional (default=0)
            Any positive number for verbosity.

        ignore_warnings: bool, optional (default=True)
            When set to True, warnings related to algorithms that were not
            run are ignored.

        custom_metric: function, optional (default=None)
            When function is provided, models are evaluated based on the
            custom evaluation metric provided.

        predictions: bool, optional (default=False)
            When set to True, the predictions of all the models models are
            returned as data frame.

        sort_by: string, optional (default='Accuracy')
            Sort models by a metric. Available options are 'Accuracy',
            'Balanced Accuracy', 'ROC AUC', 'F1 Score' or a custom metric
            identified by its name and provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators names or just 'all' for > 90
            classifiers (default='all')

        preprocess: bool, preprocessing is done when set to True

        n_jobs: int, when possible, run in parallel
            For now, only used by individual models that support it.

        All the other parameters are the same as CustomClassifier's.

    Examples:

        ```python
        import nnetsauce as ns
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        data = load_breast_cancer()
        X = data.data
        y= data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
        random_state=123)
        clf = ns.LazyClassifier(verbose=0, ignore_warnings=True)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
        print(models)
        ```

    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        sort_by="Accuracy",
        random_state=42,
        estimators="all",
        preprocess=False,
        n_jobs=None,
        # CustomClassifier attributes
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
        self.models = {}
        self.random_state = random_state
        self.estimators = estimators
        self.preprocess = preprocess
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
        """Fit classifiers to X_train and y_train, predict and score on X_test,
        y_test.

        Parameters:

            X_train: array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            X_test: array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

            y_train: array-like,
                Training vectors, where rows is the number of samples
                and columns is the number of features.

            y_test: array-like,
                Testing vectors, where rows is the number of samples
                and columns is the number of features.

        Returns:

            scores: Pandas DataFrame
                Returns metrics of all the models in a Pandas DataFrame.

            predictions: Pandas DataFrame
                Returns predictions of all the models in a Pandas DataFrame.
        """
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric is not None:
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

        if self.estimators == "all":

            self.classifiers = (
                CLASSIFIERS + MULTITASKCLASSIFIERS + SIMPLEMULTITASKCLASSIFIERS
            )

        else:  # list custom estimators, by their names

            self.classifiers = (
                [
                    ("CustomClassifier(" + est[0] + ")", est[1])
                    for est in all_estimators()
                    if (
                        issubclass(est[1], ClassifierMixin)
                        and (est[0] in self.estimators)
                    )
                ]
                + [
                    (
                        "MultitaskClassifier(" + est[0] + ")",
                        partial(MultitaskClassifier, obj=est[1]()),
                    )
                    for est in all_estimators()
                    if (
                        issubclass(est[1], RegressorMixin)
                        and (est[0] in self.estimators)
                    )
                ]
                + [
                    (
                        "SimpleMultitaskClassifier(" + est[0] + ")",
                        partial(SimpleMultitaskClassifier, obj=est[1]()),
                    )
                    for est in all_estimators()
                    if (
                        issubclass(est[1], RegressorMixin)
                        and (est[0] in self.estimators)
                    )
                ]
            )

        if self.preprocess is True:

            for name, model in tqdm(self.classifiers):  # do parallel exec

                other_args = (
                    {}
                )  # use this trick for `random_state` too --> refactor
                try:
                    if (
                        "n_jobs" in model().get_params().keys()
                        and name.find("LogisticRegression") == -1
                    ):
                        other_args["n_jobs"] = self.n_jobs
                except Exception:
                    pass

                start = time.time()

                try:
                    if "random_state" in model().get_params().keys():
                        pipe = Pipeline(
                            [
                                ("preprocessor", preprocessor),
                                (
                                    "classifier",
                                    CustomClassifier(
                                        obj=model(
                                            random_state=self.random_state,
                                            **other_args
                                        ),
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
                                    ),
                                ),
                            ]
                        )
                    else:
                        pipe = Pipeline(
                            [
                                ("preprocessor", preprocessor),
                                (
                                    "classifier",
                                    CustomClassifier(
                                        obj=model(**other_args),
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
                                    ),
                                ),
                            ]
                        )

                    pipe.fit(X_train, y_train)
                    self.models[name] = pipe
                    y_pred = pipe.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred, normalize=True)
                    b_accuracy = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred)
                    except Exception as exception:
                        roc_auc = None
                        if self.ignore_warnings is False:
                            print("ROC AUC couldn't be calculated for " + name)
                            print(exception)
                    names.append(name)
                    Accuracy.append(accuracy)
                    B_Accuracy.append(b_accuracy)
                    ROC_AUC.append(roc_auc)
                    F1.append(f1)
                    TIME.append(time.time() - start)
                    if self.custom_metric is not None:
                        custom_metric = self.custom_metric(y_test, y_pred)
                        CUSTOM_METRIC.append(custom_metric)
                    if self.verbose > 0:
                        if self.custom_metric is not None:
                            print(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy,
                                    "Balanced Accuracy": b_accuracy,
                                    "ROC AUC": roc_auc,
                                    "F1 Score": f1,
                                    self.custom_metric.__name__: custom_metric,
                                    "Time taken": time.time() - start,
                                }
                            )
                        else:
                            print(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy,
                                    "Balanced Accuracy": b_accuracy,
                                    "ROC AUC": roc_auc,
                                    "F1 Score": f1,
                                    "Time taken": time.time() - start,
                                }
                            )
                    if self.predictions:
                        predictions[name] = y_pred
                except Exception as exception:
                    try:
                        if self.ignore_warnings is False:
                            print(name + " model failed to execute")
                            print(exception)
                    except Exception as exception:
                        pass

        else:  # if self.preprocess is False:

            for name, model in tqdm(self.classifiers):  # do parallel exec
                other_args = (
                    {}
                )  # use this trick for `random_state` too --> refactor
                try:
                    if (
                        "n_jobs" in model().get_params().keys()
                        and name.find("LogisticRegression") == -1
                    ):
                        other_args["n_jobs"] = self.n_jobs
                except Exception:
                    pass

                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        pipe = CustomClassifier(
                            obj=model(
                                random_state=self.random_state, **other_args
                            ),
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
                        pipe = CustomClassifier(
                            obj=model(**other_args),
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

                    pipe.fit(X_train, y_train)
                    self.models[name] = pipe
                    y_pred = pipe.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred, normalize=True)
                    b_accuracy = balanced_accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="weighted")
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred)
                    except Exception as exception:
                        roc_auc = None
                        if self.ignore_warnings is False:
                            print("ROC AUC couldn't be calculated for " + name)
                            print(exception)
                    names.append(name)
                    Accuracy.append(accuracy)
                    B_Accuracy.append(b_accuracy)
                    ROC_AUC.append(roc_auc)
                    F1.append(f1)
                    TIME.append(time.time() - start)
                    if self.custom_metric is not None:
                        custom_metric = self.custom_metric(y_test, y_pred)
                        CUSTOM_METRIC.append(custom_metric)
                    if self.verbose > 0:
                        if self.custom_metric is not None:
                            print(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy,
                                    "Balanced Accuracy": b_accuracy,
                                    "ROC AUC": roc_auc,
                                    "F1 Score": f1,
                                    self.custom_metric.__name__: custom_metric,
                                    "Time taken": time.time() - start,
                                }
                            )
                        else:
                            print(
                                {
                                    "Model": name,
                                    "Accuracy": accuracy,
                                    "Balanced Accuracy": b_accuracy,
                                    "ROC AUC": roc_auc,
                                    "F1 Score": f1,
                                    "Time taken": time.time() - start,
                                }
                            )
                    if self.predictions:
                        predictions[name] = y_pred
                except Exception as exception:
                    try:
                        if self.ignore_warnings is False:
                            print(name + " model failed to execute")
                            print(exception)
                    except Exception as exception:
                        pass

        if self.custom_metric is None:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Time Taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    self.custom_metric.__name__: CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )
        scores = scores.sort_values(by=self.sort_by, ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """Returns all the model objects trained. If fit hasn't been called yet,
        then it's called to return the models.

        Parameters:

        X_train: array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.

        X_test: array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.

        y_train: array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.

        y_test: array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.

        Returns:

            models: dict-object,
                Returns a dictionary with each model's pipeline as value
                and key = name of the model.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models
