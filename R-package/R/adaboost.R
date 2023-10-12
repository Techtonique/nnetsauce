
# 1 - Classifiers ---------------------------------------------------------

#' Adaboost classifier with quasi-randomized hidden layer
#'
#' Parameters' description can be found at \url{https://techtonique.github.io/nnetsauce/}
#'
#' @return
#' @export
#'
#' @examples
#'
#' library(datasets)
#'
#' X <- as.matrix(iris[, 1:4])
#' y <- as.integer(iris[, 5]) - 1L
#'
#' n <- dim(X)[1]
#' p <- dim(X)[2]
#'
#' set.seed(213)
#' train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
#' test_index <- -train_index
#'
#' X_train <- as.matrix(iris[train_index, 1:4])
#' y_train <- as.integer(iris[train_index, 5]) - 1L
#' X_test <- as.matrix(iris[test_index, 1:4])
#' y_test <- as.integer(iris[test_index, 5]) - 1L
#'
#' # ValueError: Sample weights must be 1D array or scalar
#' # obj <- sklearn$tree$DecisionTreeClassifier()
#' # obj2 <- AdaBoostClassifier(obj)
#' # obj2$fit(X_train, y_train)
#' # print(obj2$score(X_test, y_test))
#' # print(obj2$predict_proba(X_test))
#'
AdaBoostClassifier <- function(obj,
                              n_estimators=10L,
                              learning_rate=0.1,
                              n_hidden_features=1L,
                              reg_lambda=0,
                              reg_alpha=0.5,
                              activation_name="relu",
                              a=0.01,
                              nodes_sim="sobol",
                              bias=TRUE,
                              dropout=0,
                              direct_link=FALSE,
                              n_clusters=2L,
                              cluster_encode=TRUE,
                              type_clust="kmeans",
                              col_sample=1,
                              row_sample=1,
                              seed=123L,
                              verbose=1,
                              method="SAMME",
                              backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

  ns$AdaBoostClassifier(obj,
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        n_hidden_features=n_hidden_features,
                        reg_lambda=reg_lambda,
                        reg_alpha=reg_alpha,
                        activation_name=activation_name,
                        a=a,
                        nodes_sim=nodes_sim,
                        bias=bias,
                        dropout=dropout,
                        direct_link=direct_link,
                        n_clusters=n_clusters,
                        cluster_encode=cluster_encode,
                        type_clust=type_clust,
                        col_sample=col_sample,
                        row_sample=row_sample,
                        seed=seed,
                        verbose=verbose,
                        method=method,
                        backend=backend)
}

# 2 - Regressors ---------------------------------------------------------
