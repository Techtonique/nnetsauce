
# 1 - Classifiers ---------------------------------------------------------

#' Bootstrap aggregating with quasi-randomized layer (classification)
#'
#' Parameters description can be found at \url{https://techtonique.github.io/nnetsauce/}
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
#' obj <- sklearn$tree$DecisionTreeClassifier()
#' obj2 <- RandomBagClassifier(obj, n_estimators=50L,
#' n_hidden_features=5L)
#' obj2$fit(X_train, y_train)
#' print(obj2$score(X_test, y_test))
#' print(obj2$predict_proba(X_test))
#'
RandomBagClassifier <- function(obj,
                                n_estimators=50L,
                                n_hidden_features=5L,
                                activation_name="relu",
                                a=0.01,
                                nodes_sim="sobol",
                                bias=TRUE,
                                dropout=0,
                                direct_link=FALSE,
                                n_clusters=2L,
                                cluster_encode=TRUE,
                                type_clust="kmeans",
                                col_sample=1,
                                row_sample=1,
                                n_jobs=NULL,
                                seed=123L,
                                verbose=1L,
                                backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

  ns$RandomBagClassifier(obj=obj,
                         n_estimators=n_estimators,
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
                         n_jobs=n_jobs,
                         seed=seed,
                         verbose=verbose,
                         backend=backend)
}

# 2 - Regressors ---------------------------------------------------------

#' Bootstrap aggregating with quasi-randomized layer (regression)
#'
#' Parameters description can be found at \url{https://techtonique.github.io/nnetsauce/}
#'
#' @return
#' @export
#'
#' @examples
#'
#' library(datasets)
#'
#' n <- 20 ; p <- 5
#' X <- matrix(rnorm(n * p), n, p) # no intercept!
#' y <- rnorm(n)
#'
#' obj <- sklearn$tree$DecisionTreeRegressor()
#' obj2 <- RandomBagRegressor(obj)
#' obj2$fit(X[1:12,], y[1:12])
#' print(obj2$score(X[13:20, ], y[13:20]))
#'
RandomBagRegressor <- function(obj,
                                n_estimators=10L,
                                n_hidden_features=1L,
                                activation_name="relu",
                                a=0.01,
                                nodes_sim="sobol",
                                bias=TRUE,
                                dropout=0,
                                direct_link=FALSE,
                                n_clusters=2L,
                                cluster_encode=TRUE,
                                type_clust="kmeans",
                                col_sample=1,
                                row_sample=1,
                                n_jobs=NULL,
                                seed=123L,
                                verbose=1L,
                                backend=c("cpu", "gpu", "tpu"))
{
  backend <- match.arg(backend)

  ns$RandomBagRegressor(obj=obj,
                         n_estimators=n_estimators,
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
                         n_jobs=n_jobs,
                         seed=seed,
                         verbose=verbose,
                         backend=backend)
}
