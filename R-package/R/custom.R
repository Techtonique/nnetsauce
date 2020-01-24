
# 1 - Classifiers ---------------------------------------------------------


#' Custom classifier with quasi-randomized layer
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
#' @param col_sample
#' @param row_sample
#' @param seed
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
#' obj2 <- CustomClassifier(obj)
#' obj2$fit(X, y)
#' print(obj2$score(X, y))
#'
CustomClassifier <- function(obj,
                             n_hidden_features=5L,
                             activation_name="relu",
                             a=0.01,
                             nodes_sim="sobol",
                             bias=TRUE,
                             dropout=0,
                             direct_link=TRUE,
                             n_clusters=2L,
                             cluster_encode=TRUE,
                             type_clust="kmeans",
                             col_sample=1,
                             row_sample=1,
                             seed=123L)
{
  ns$CustomClassifier(obj,
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
                      seed=seed)
}


# 2 - Regressors ---------------------------------------------------------


#'  Custom regressor with quasi-randomized layer
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
#' @param col_sample
#' @param row_sample
#' @param seed
#'
#' @return
#' @export
#'
#' @examples
#'
#' set.seed(123)
#' n <- 50 ; p <- 3
#' X <- matrix(rnorm(n * p), n, p) # no intercept!
#' y <- rnorm(n)
#'
#' obj <- sklearn$linear_model$ElasticNet()
#' obj2 <- CustomRegressor(obj)
#' obj2$fit(X, y)
#' print(obj2$score(X, y))
#'
CustomRegressor <- function(obj,
                             n_hidden_features=5L,
                             activation_name="relu",
                             a=0.01,
                             nodes_sim="sobol",
                             bias=TRUE,
                             dropout=0,
                             direct_link=TRUE,
                             n_clusters=2L,
                             cluster_encode=TRUE,
                             type_clust="kmeans",
                             col_sample=1,
                             row_sample=1,
                             seed=123L)
{
  ns$CustomRegressor(obj,
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
                      seed=seed)
}
