
# 1 - Classifiers ---------------------------------------------------------

# 2 - Regressors ---------------------------------------------------------

#' Bayesian Random Vector Functional link network with 1 shrinkage parameter
#'
#' Parameters' description can be found at \url{https://techtonique.github.io/nnetsauce/}
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
#' (index_train <- base::sample.int(n = nrow(X),
#'                                  size = floor(0.8*nrow(X)),
#'                                  replace = FALSE))
#' X_train <- X[index_train, ]
#' y_train <- y[index_train]
#' X_test <- X[-index_train, ]
#' y_test <- y[-index_train]
#'
#' obj <- BayesianRVFLRegressor(n_hidden_features = 5L)
#' print(obj$fit(X_train, y_train))
#' print(obj$score(X_test, y_test))
#'
BayesianRVFLRegressor <- function(n_hidden_features=5L,
                                  activation_name="relu",
                                  a=0.01,
                                  nodes_sim="sobol",
                                  bias=TRUE,
                                  dropout=0,
                                  direct_link=TRUE,
                                  n_clusters=2L,
                                  cluster_encode=TRUE,
                                  type_clust="kmeans",
                                  s=0.1,
                                  sigma=0.05,
                                  seed=123L,
                                  backend=c("cpu", "gpu", "tpu")
                                  )
{
  backend <- match.arg(backend)

  ns$BayesianRVFLRegressor(n_hidden_features=n_hidden_features,
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
                           s=s,
                           sigma=sigma,
                           backend=backend)
}


#' Bayesian Random Vector Functional link network with 2 shrinkage parameters
#'
#' Parameters' description can be found at \url{https://techtonique.github.io/nnetsauce/}
#'#' @return
#' @export
#'
#' @examples
#'
#' set.seed(123)
#' n <- 50 ; p <- 3
#' X <- matrix(rnorm(n * p), n, p) # no intercept!
#' y <- rnorm(n)
#'
#' (index_train <- base::sample.int(n = nrow(X),
#'                                  size = floor(0.8*nrow(X)),
#'                                  replace = FALSE))
#' X_train <- X[index_train, ]
#' y_train <- y[index_train]
#' X_test <- X[-index_train, ]
#' y_test <- y[-index_train]
#'
#' obj <- BayesianRVFL2Regressor(n_hidden_features = 5L, s1=0.01)
#' print(obj$fit(X_train, y_train))
#' print(obj$score(X_test, y_test))
#'
BayesianRVFL2Regressor <- function(n_hidden_features=5L,
                                   activation_name="relu",
                                   a=0.01,
                                   nodes_sim="sobol",
                                   bias=TRUE,
                                   dropout=0,
                                   direct_link=TRUE,
                                   n_clusters=2L,
                                   cluster_encode=TRUE,
                                   type_clust="kmeans",
                                   s1=0.1,
                                   s2=0.1,
                                   sigma=0.05,
                                   seed=123L,
                                   backend=c("cpu", "gpu", "tpu")
                                   )
{
  backend <- match.arg(backend)

 ns$BayesianRVFL2Regressor(n_hidden_features=n_hidden_features,
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
                           s1=s1,
                           s2=s2,
                           sigma=sigma,
                           backend=backend)
}
