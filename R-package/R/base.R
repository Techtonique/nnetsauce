
# 1 - Classifiers ---------------------------------------------------------

# 2 - Regressors ---------------------------------------------------------


#' Linear regressor with a quasi-randomized layer
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
#' n <- dim(X)[1]
#' p <- dim(X)[2]
#'
#' set.seed(213)
#' train_index <- sample(x = 1:n, size = floor(0.8*n), replace = TRUE)
#' test_index <- -train_index
#'
#' X_train <- as.matrix(X[train_index, ])
#' y_train <- y[train_index]
#' X_test <- as.matrix(X[test_index, ])
#' y_test <- y[test_index]
#'
#' obj <- BaseRegressor(n_hidden_features=10L, dropout=0.9)
#' print(obj$fit(X_train, y_train))
#' print(obj$score(X_test, y_test))
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
                          backend=c("cpu", "gpu", "tpu"))
{
 backend <- match.arg(backend)

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
