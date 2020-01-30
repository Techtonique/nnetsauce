
# 1 - Classifiers ---------------------------------------------------------

#' Adaboost classifier with quasi-randomized hidden layer
#'
#' @param obj
#' @param n_estimators
#' @param learning_rate
#' @param n_hidden_features
#' @param reg_lambda
#' @param reg_alpha
#' @param activation_name
#' @param a
#' @param nodes_sim
#' @param bias
#' @param dropout
#' @param direct_link
#' @param n_clusters
#' @param cluster_encode
#' @param type_clust
#' @param col_sample
#' @param row_sample
#' @param seed
#' @param verbose
#' @param method
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
#' obj <- sklearn$tree$DecisionTreeClassifier()
#' obj2 <- AdaBoostClassifier(obj)
#' # obj2$fit(X, y)
#' # print(obj2$score(X, y))
#' # print(obj2$predict_proba(X))
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
                              method="SAMME")
{
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
                        method=method)
}

# 2 - Regressors ---------------------------------------------------------
