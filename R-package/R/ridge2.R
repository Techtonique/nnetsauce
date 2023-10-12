
# 1 - Classifiers ---------------------------------------------------------

#' Multinomial logit, quasi-randomized classification model with 2 shrinkage parameters
#'
#' Parameters' description can be found at \url{https://techtonique.github.io/nnetsauce/}
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
#' (index_train <- base::sample.int(n = nrow(X),
#'                                  size = floor(0.8*nrow(X)),
#'                                  replace = FALSE))
#' X_train <- X[index_train, ]
#' y_train <- y[index_train]
#' X_test <- X[-index_train, ]
#' y_test <- y[-index_train]
#'
#' obj <- Ridge2Classifier()
#' obj$fit(X_train, y_train)
#' print(obj$score(X_test, y_test))
#' print(obj$predict_proba(X_train))
#'
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
                             type_clust = "kmeans", # type_scaling
                             lambda1 = 0.1,
                             lambda2 = 0.1,
                             seed = 123L,
                             backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

  ns$Ridge2Classifier(
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
    lambda1 = lambda1,
    lambda2 = lambda2,
    seed = seed,
    backend=backend
  )
}


#' Multitask quasi-randomized classification model with 2 shrinkage parameters
#'
#' Parameters' description can be found at \url{https://techtonique.github.io/nnetsauce/}
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
#' (index_train <- base::sample.int(n = nrow(X),
#'                                  size = floor(0.8*nrow(X)),
#'                                  replace = FALSE))
#' X_train <- X[index_train, ]
#' y_train <- y[index_train]
#' X_test <- X[-index_train, ]
#' y_test <- y[-index_train]
#'
#' obj <- Ridge2MultitaskClassifier()
#' obj$fit(X_train, y_train)
#' print(obj$score(X_test, y_test))
#' print(obj$predict_proba(X_train))
#'
#'
Ridge2MultitaskClassifier <- function(n_hidden_features=5L,
                                  activation_name="relu",
                                  a=0.01,
                                  nodes_sim="sobol",
                                  bias=TRUE,
                                  dropout=0,
                                  n_clusters=2L,
                                  cluster_encode=TRUE,
                                  type_clust="kmeans", # type_scaling
                                  lambda1=0.1,
                                  lambda2=0.1,
                                  seed=123L,
                                  backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

  ns$Ridge2MultitaskClassifier(n_hidden_features=n_hidden_features,
                           activation_name=activation_name,
                           a=a,
                           nodes_sim=nodes_sim,
                           bias=bias,
                           dropout=dropout,
                           n_clusters=n_clusters,
                           cluster_encode=cluster_encode,
                           type_clust=type_clust, # type_scaling
                           lambda1=lambda1,
                           lambda2=lambda2,
                           seed=seed,
                           backend=backend)
}


# 2 - Regressors ---------------------------------------------------------


#' Quasi-randomized regression model with 2 shrinkage parameters
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
#' obj <- nnetsauce::Ridge2Regressor(n_hidden_features = 5L)
#' print(obj$fit(X, y))
#' print(obj$score(X, y))
#'
Ridge2Regressor <- function(n_hidden_features=5L,
                            activation_name="relu",
                            a=0.01,
                            nodes_sim="sobol",
                            bias=TRUE,
                            dropout=0,
                            n_clusters=2L,
                            cluster_encode=TRUE,
                            type_clust="kmeans", # type_scaling
                            lambda1=0.1,
                            lambda2=0.1,
                            seed=123L,
                            backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

  ns$Ridge2Regressor(n_hidden_features=n_hidden_features,
                     activation_name=activation_name,
                     a=a,
                     nodes_sim=nodes_sim,
                     bias=bias,
                     dropout=dropout,
                     n_clusters=n_clusters,
                     cluster_encode=cluster_encode,
                     type_clust=type_clust, # type_scaling
                     lambda1=lambda1,
                     lambda2=lambda2,
                     seed=seed,
                     backend=backend)
}
