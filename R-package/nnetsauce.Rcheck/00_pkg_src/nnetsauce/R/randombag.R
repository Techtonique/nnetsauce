
# 1 - Classifiers ---------------------------------------------------------

#' Bootstrap aggregating with quasi-randomized layer
#'
#' @param obj
#' @param n_estimators
#' @param n_hidden_features
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
#' @param n_jobs
#' @param seed
#' @param verbose
#' @param backend
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
#' obj2 <- RandomBagClassifier(obj)
#' obj2$fit(X, y)
#' print(obj2$score(X, y))
#' print(obj2$predict_proba(X))
#'
RandomBagClassifier <- function(obj,
                                n_estimators=10L,
                                n_hidden_features=1L,
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
                                n_jobs=NULL,
                                seed=123L,
                                verbose=1L,
                                backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

  ns$RandomBagClassifier(obj=obj,
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
                         col_sample=col_sample,
                         row_sample=row_sample,
                         n_jobs=n_jobs,
                         seed=seed,
                         verbose=verbose,
                         backend=backend)
}

# 2 - Regressors ---------------------------------------------------------

