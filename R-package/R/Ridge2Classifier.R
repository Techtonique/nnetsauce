#' Title
#'
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
#' @param lambda1
#' @param lambda2
#' @param seed
#'
#' @return
#' @export
#'
#' @examples
#'
#'
#' # Example 1 -----
#'
#' library(datasets)
#'
#' X <- as.matrix(iris[, 1:4])
#' y <- as.integer(iris[, 5]) - 1L
#'
#' obj <- Ridge2Classifier()
#' obj$fit(X, y)
#' print(obj$score(X, y))
#' print(obj$predict_proba(X))
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
#' obj2 <- Ridge2Classifier()
#' obj2$fit(X, y)
#' print(obj2$score(X, y))
#' print(obj2$predict_proba(X))
#'
Ridge2Classifier <- function(n_hidden_features = 5L,
                             activation_name = "relu",
                             a = 0.01,
                             nodes_sim = "sobol",
                             bias = TRUE,
                             dropout = 0,
                             direct_link = TRUE,
                             n_clusters = 2L,
                             cluster_encode = TRUE,
                             type_clust = "kmeans",
                             col_sample = 1,
                             row_sample = 1,
                             lambda1 = 0.1,
                             lambda2 = 0.1,
                             seed = 123L)
{
  return(ns$Ridge2Classifier(
    n_hidden_features = n_hidden_features,
    activation_name = activation_name,
    a = a,
    nodes_sim = nodes_sim,
    bias = bias,
    dropout = dropout,
    direct_link = direct_link,
    n_clusters = n_clusters,
    cluster_encode = cluster_encode,
    type_clust = type_clust,
    col_sample = col_sample,
    row_sample = row_sample,
    lambda1 = lambda1,
    lambda2 = lambda2,
    seed = seed
  ))
}
