#' Generalized nonlinears models for Classification
#'
#' @param n_hidden_features
#' @param lambda1
#' @param alpha1
#' @param lambda2
#' @param alpha2
#' @param family
#' @param activation_name
#' @param a
#' @param nodes_sim
#' @param bias
#' @param dropout
#' @param direct_link
#' @param n_clusters
#' @param cluster_encode
#' @param type_clust
#' @param type_scaling
#' @param seed
#'
#' @return
#' @export
#'
#' @examples
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

#' Generalized nonlinears models for continuous output (regression)
#'
#' @param n_hidden_features
#' @param lambda1
#' @param alpha1
#' @param lambda2
#' @param alpha2
#' @param family
#' @param activation_name
#' @param a
#' @param nodes_sim
#' @param bias
#' @param dropout
#' @param direct_link
#' @param n_clusters
#' @param cluster_encode
#' @param type_clust
#' @param type_scaling
#' @param seed
#'
#' @return
#' @export
#'
#' @examples
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

