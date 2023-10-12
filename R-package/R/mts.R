#' Multivariate Time Series
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
#' X <- matrix(rnorm(300), 100, 3)
#'
#' obj <- sklearn$linear_model$BayesianRidge()
#' obj2 <- MTS(obj)
#'
#' obj2$fit(X)
#' obj2$predict(return_std = TRUE)
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
                seed=123L,
                lags=1L,
                backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

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
         seed=seed,
         lags=lags,
         backend=backend)
}
