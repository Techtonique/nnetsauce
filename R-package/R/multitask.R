
# 1 - Classifiers ---------------------------------------------------------

#' Multitask Classification model based on regression models, with shared covariates
#'
#' Parameters description can be found at \url{https://techtonique.github.io/nnetsauce/}
#'
#' @return
#' @export
#'
#' @examples
#'
#' # Example 1 -----
#'
#' library(datasets)
#'
#' X <- as.matrix(iris[, 1:4])
#' y <- as.integer(iris[, 5]) - 1L
#'
#' obj <- sklearn$linear_model$LinearRegression()
#' obj2 <- MultitaskClassifier(obj)
#' obj2$fit(X, y)
#' print(obj2$score(X, y))
#' print(obj2$predict_proba(X))
#'
#'
#' # Example 2 -----
#'
#' n <- 25
#' p <- 4
#
#' set.seed(123)
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- sample(c(0L, 1L), n, replace = TRUE)
#'
#' obj <- sklearn$linear_model$LinearRegression()
#' obj2 <- MultitaskClassifier(obj)
#' obj2$fit(X, y)
#' print(obj2$score(X, y))
#' print(obj2$predict_proba(X))
#'
#'
MultitaskClassifier <- function(obj,
                                n_hidden_features=5L,
                                activation_name="relu",
                                a=0.01,
                                nodes_sim="sobol",
                                bias=TRUE,
                                dropout=0,
                                direct_link=TRUE,
                                n_clusters=2L,
                                cluster_encode=TRUE,
                                type_clust="kmeans", # type_scaling
                                col_sample=1,
                                row_sample=1,
                                seed=123L,
                                backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

  ns$MultitaskClassifier(obj=obj,
                         n_hidden_features=n_hidden_features,
                         activation_name=activation_name,
                         a=a,
                         nodes_sim=nodes_sim,
                         bias=bias,
                         dropout=dropout,
                         direct_link=direct_link,
                         n_clusters=n_clusters,
                         cluster_encode=cluster_encode,
                         type_clust=type_clust, # type_scaling
                         col_sample=col_sample,
                         row_sample=row_sample,
                         seed=seed,
                         backend=backend)
}
