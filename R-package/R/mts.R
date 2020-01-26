#' Multivariate Time Series
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
#' @param seed
#' @param lags
#' @param return_std
#'
#' @return
#' @export
#'
#' @examples
#'
#' # Example 1 -----
#'
#' set.seed(123)
#' X <- matrix(rnorm(300), 100, 3)
#'
#' obj <- sklearn$linear_model$ElasticNet()
#' obj2 <- MTS(obj)
#'
#' obj2$fit(X)
#' obj2$predict()
#'
#'
#' # Example 2 -----
#'
#' set.seed(123)
#' X <- matrix(rnorm(300), 100, 2)
#'
#' obj <- sklearn$linear_model$BayesianRidge()
#' obj2 <- MTS(obj)
#'
#' obj2$fit(X)
#' obj2$predict(return_std=TRUE)
#'
MTS <- function(obj,
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
                seed=123L,
                lags=1L,
                return_std=FALSE)
{
  ns$MTS(obj,
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
         seed=seed,
         lags=lags,
         return_std=return_std)
}
