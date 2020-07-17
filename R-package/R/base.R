
# 1 - Classifiers ---------------------------------------------------------

# 2 - Regressors ---------------------------------------------------------


#' Linear regressor with a quasi-randomized layer
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
#' @param seed
#' @param backend
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
#' obj <- nnetsauce::BaseRegressor(n_hidden_features=10L, dropout=0.9)
#' print(obj$fit(X, y))
#' print(obj$score(X, y))
#'
BaseRegressor <- function(n_hidden_features=5L,
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
                          seed=123L, 
                          backend="cpu")
{
 ns$BaseRegressor(n_hidden_features=n_hidden_features,
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
                  backend=backend)
}
