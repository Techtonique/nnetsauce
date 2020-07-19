
# 1 - Classifiers ---------------------------------------------------------

#' Multitask Classification model based on regression models, with shared covariates
#'
#' @param obj
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
#' @param type_scaling
#' @param col_sample
#' @param row_sample
#' @param seed
#' @param backend
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
  if ((as.character(Sys.info()[1])=="Windows") && (backend %in% c("gpu", "tpu")))
  {
      warning("No GPU/TPU computing on Windows yet, backend set to 'cpu'")
      backend <- "cpu"  
  }     
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
