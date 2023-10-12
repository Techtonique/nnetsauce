#' Generalized nonlinear models for Classification
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
#' set.seed(123)
#' X <- as.matrix(iris[, 1:4])
#' y <- as.integer(iris$Species) - 1L
#'
#' (index_train <- base::sample.int(n = nrow(X),
#'                                  size = floor(0.8*nrow(X)),
#'                                  replace = FALSE))
#' X_train <- X[index_train, ]
#' y_train <- y[index_train]
#' X_test <- X[-index_train, ]
#' y_test <- y[-index_train]
#'
#' obj <- GLMClassifier()
#' obj$fit(X_train, y_train)
#' print(obj$score(X_test, y_test))
#'
GLMClassifier <- function(n_hidden_features=5L,
                          lambda1=0.01,
                          alpha1=0.5,
                          lambda2=0.01,
                          alpha2=0.5,
                          family="expit",
                          activation_name="relu",
                          a=0.01,
                          nodes_sim="sobol",
                          bias=TRUE,
                          dropout=0,
                          direct_link=TRUE,
                          n_clusters=2L,
                          cluster_encode=TRUE,
                          type_clust="kmeans",
                          type_scaling=c("std", "std", "std"),
                          optimizer=ns$Optimizer(),
                          seed=123L)
{
  #backend <- match.arg(backend)

  ns$GLMClassifier(n_hidden_features=n_hidden_features,
                   lambda1=lambda1,
                   alpha1=alpha1,
                   lambda2=lambda2,
                   alpha2=alpha2,
                   family=family,
                   activation_name=activation_name,
                   a=a,
                   nodes_sim=nodes_sim,
                   bias=bias,
                   dropout=dropout,
                   direct_link=direct_link,
                   n_clusters=n_clusters,
                   cluster_encode=cluster_encode,
                   type_clust=type_clust,
                   type_scaling=type_scaling,
                   optimizer=optimizer,
                   seed=seed)
}

#' Generalized nonlinear models for continuous output (regression)
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
#' obj <- GLMRegressor()
#' obj$fit(X_train, y_train)
#' print(obj$score(X_test, y_test))
#'
GLMRegressor <- function(n_hidden_features=5L,
                          lambda1=0.01,
                          alpha1=0.5,
                          lambda2=0.01,
                          alpha2=0.5,
                          family="gaussian",
                          activation_name="relu",
                          a=0.01,
                          nodes_sim="sobol",
                          bias=TRUE,
                          dropout=0,
                          direct_link=TRUE,
                          n_clusters=2L,
                          cluster_encode=TRUE,
                          type_clust="kmeans",
                          type_scaling=c("std", "std", "std"),
                          optimizer=ns$Optimizer(),
                          seed=123L)
{
  # backend <- match.arg(backend)

  ns$GLMRegressor(n_hidden_features=n_hidden_features,
                   lambda1=lambda1,
                   alpha1=alpha1,
                   lambda2=lambda2,
                   alpha2=alpha2,
                   family=family,
                   activation_name=activation_name,
                   a=a,
                   nodes_sim=nodes_sim,
                   bias=bias,
                   dropout=dropout,
                   direct_link=direct_link,
                   n_clusters=n_clusters,
                   cluster_encode=cluster_encode,
                   type_clust=type_clust,
                   type_scaling=type_scaling,
                   optimizer=optimizer,
                   seed=seed)
}

