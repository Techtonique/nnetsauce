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
                start_input = NULL,
                frequency_input = NULL,
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
                lags=1L,
                replications=NULL,
                kernel=NULL,
                agg="mean",
                seed=123L,
                backend=c("cpu", "gpu", "tpu"),
                verbose=0)
{
  backend <- match.arg(backend)

  res <- ns$MTS(obj,
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
         lags=lags,
         replications=replications,
         kernel=kernel,
         agg=agg,
         seed=seed,
         backend=backend,
         verbose=verbose)

  res$method <- "MTS"
  res$start <- start_input
  res$frequency <- frequency_input

  return(res)
}
