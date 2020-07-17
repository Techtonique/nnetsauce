
# 1 - Classifiers ---------------------------------------------------------

#' Multinomial logit, quasi-randomized classification model with 2 shrinkage parameters
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
                             type_clust = "kmeans", # type_scaling
                             col_sample = 1,
                             row_sample = 1,
                             lambda1 = 0.1,
                             lambda2 = 0.1,
                             seed = 123L, 
                             backend="cpu")
{
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
    col_sample = col_sample,
    row_sample = row_sample,
    lambda1 = lambda1,
    lambda2 = lambda2,
    seed = seed,
    backend=backend
  )
}


#' Multitask quasi-randomized classification model with 2 shrinkage parameters
#'
#' @param n_hidden_features
#' @param activation_name
#' @param a
#' @param nodes_sim
#' @param bias
#' @param dropout
#' @param n_clusters
#' @param cluster_encode
#' @param type_clust
#' @param type_scaling
#' @param lambda1
#' @param lambda2
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
#' obj <- Ridge2MultitaskClassifier()
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
#' obj2 <- Ridge2MultitaskClassifier()
#' obj2$fit(X, y)
#' print(obj2$score(X, y))
#' print(obj2$predict_proba(X))
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
                                  backend="cpu")
{
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
#' @param n_hidden_features
#' @param activation_name
#' @param a
#' @param nodes_sim
#' @param bias
#' @param dropout
#' @param n_clusters
#' @param cluster_encode
#' @param type_clust
#' @param col_sample
#' @param row_sample
#' @param lambda1
#' @param lambda2
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
                            col_sample=1,
                            row_sample=1,
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
                     col_sample=col_sample,
                     row_sample=row_sample,
                     lambda1=lambda1,
                     lambda2=lambda2,
                     seed=seed,
                     backend=backend)
}
