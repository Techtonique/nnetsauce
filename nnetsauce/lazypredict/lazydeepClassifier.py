import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except ImportError:
    pass
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from functools import partial
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
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)
from .config import (
    DEEPCLASSIFIERS,
    DEEPMULTITASKCLASSIFIERS,
    DEEPSIMPLEMULTITASKCLASSIFIERS,
)
from ..custom import Custom, CustomClassifier
from ..multitask import MultitaskClassifier, SimpleMultitaskClassifier

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


class LazyDeepClassifier(Custom, ClassifierMixin):
    """

    Fitting -- almost -- all the classification algorithms with layers of
    nnetsauce's CustomClassifier and returning their scores.

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
            When set to True, the predictions of all the models models are
            returned as data frame.

        sort_by: string, optional (default='Accuracy')
            Sort models by a metric. Available options are 'Accuracy',
            'Balanced Accuracy', 'ROC AUC', 'F1 Score' or a custom metric
            identified by its name and provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators names or just 'all' for > 90 classifiers
            (default='all')

        preprocess: bool, preprocessing is done when set to True

        n_jobs: int, when possible, run in parallel
            For now, only used by individual models that support it.

        n_layers: int, optional (default=3)
            Number of layers of CustomClassifiers to be used.

        All the other parameters are the same as CustomClassifier's.

    Attributes:

        models_: dict-object
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        
        best_model_: object
            Returns the best model pipeline.

    Examples

        ```python
        import nnetsauce as ns
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        data = load_breast_cancer()
        X = data.data
        y= data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
            random_state=123)
        clf = ns.LazyDeepClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
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
        # Defining depth
        n_layers=3,
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
        
        # baseline models
        try: 
            baseline_names = ["RandomForestClassifier", "XGBClassifier"]
            baseline_models = [RandomForestClassifier(), xgb.XGBClassifier()]
        except Exception as exception:
            baseline_names = ["RandomForestClassifier"]
            baseline_models = [RandomForestClassifier()]

        for name, model in zip(baseline_names, baseline_models):
            start = time.time()
            try:
                model.fit(X_train, y_train)
                self.models_[name] = model
                y_pred = model.predict(X_test)
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
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        if self.estimators == "all":
            self.classifiers = [
                item
                for sublist in [
                    DEEPCLASSIFIERS,
                    DEEPMULTITASKCLASSIFIERS,
                    DEEPSIMPLEMULTITASKCLASSIFIERS,
                ]
                for item in sublist
            ]
        else:
            self.classifiers = (
                [
                    ("DeepCustomClassifier(" + est[0] + ")", est[1])
                    for est in all_estimators()
                    if (
                        issubclass(est[1], ClassifierMixin)
                        and (est[0] in self.estimators)
                    )
                ]
                + [
                    (
                        "DeepMultitaskClassifier(" + est[0] + ")",
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
                        "DeepSimpleMultitaskClassifier(" + est[0] + ")",
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
                        layer_clf = CustomClassifier(
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
                        layer_clf = CustomClassifier(
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

                    layer_clf.fit(X_train, y_train)

                    for _ in range(self.n_layers):
                        layer_clf = deepcopy(
                            CustomClassifier(
                                obj=layer_clf,
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

                    pipe = Pipeline(
                        [
                            ("preprocessor", preprocessor),
                            ("classifier", layer_clf),
                        ]
                    )

                    pipe.fit(X_train, y_train)
                    self.models_[name] = pipe
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
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)

        else:  # no preprocessing

            for name, model in tqdm(self.classifiers):  # do parallel exec
                start = time.time()
                try:
                    if "random_state" in model().get_params().keys():
                        layer_clf = CustomClassifier(
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
                        layer_clf = CustomClassifier(
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

                    layer_clf.fit(X_train, y_train)

                    for _ in range(self.n_layers):
                        layer_clf = deepcopy(
                            CustomClassifier(
                                obj=layer_clf,
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

                        # layer_clf.fit(X_train, y_train)

                    layer_clf.fit(X_train, y_train)

                    self.models_[name] = layer_clf
                    y_pred = layer_clf.predict(X_test)
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
                    if self.ignore_warnings is False:
                        print(name + " model failed to execute")
                        print(exception)

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
                    "Custom metric": CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )
        scores = scores.sort_values(by=self.sort_by, ascending=False).set_index(
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
        if len(self.models_.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models_


class LazyClassifier(LazyDeepClassifier):
    """
        Fitting -- almost -- all the classification algorithms with 
        nnetsauce's CustomClassifier and returning their scores (no layers).

    Parameters:

        verbose: int, optional (default=0)
            Any positive number for verbosity.

        ignore_warnings: bool, optional (default=True)
            When set to True, the warning related to algorigms that are not able to run are ignored.

        custom_metric: function, optional (default=None)
            When function is provided, models are evaluated based on the custom evaluation metric provided.

        predictions: bool, optional (default=False)
            When set to True, the predictions of all the models models are returned as dataframe.

        sort_by: string, optional (default='Accuracy')
            Sort models by a metric. Available options are 'Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1 Score'
            or a custom metric identified by its name and provided by custom_metric.

        random_state: int, optional (default=42)
            Reproducibiility seed.

        estimators: list, optional (default='all')
            list of Estimators names or just 'all' (default='all')

        preprocess: bool
            preprocessing is done when set to True

        n_jobs : int, when possible, run in parallel
            For now, only used by individual models that support it.

        All the other parameters are the same as CustomClassifier's.
    
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

        dataset = datasets.load_iris()
        X = dataset.data
        y = dataset.target
        X, y = shuffle(X, y, random_state=123)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X_train, X_test = X[:100], X[100:]
        y_train, y_test = y[:100], y[100:]

        clf = ns.LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
        print(models)
    
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