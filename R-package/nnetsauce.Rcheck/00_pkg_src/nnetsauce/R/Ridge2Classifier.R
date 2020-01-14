# obj <- nnetsauce::nnnetsauce$Ridge2Classifier(n_hidden_features=5,
#                                       activation_name="relu",
#                                       a=0.01,
#                                       nodes_sim="sobol",
#                                       bias=True,
#                                       dropout=0,
#                                       direct_link=True,
#                                       n_clusters=2,
#                                       cluster_encode=True,
#                                       type_clust="kmeans",
#                                       col_sample=1,
#                                       row_sample=1,
#                                       lambda1=0.1,
#                                       lambda2=0.1,
#                                       seed=123)
# print(obj$get_params(obj))
# # nnetsauce::sklearn$tree$DecisionTreeClassifier()
# # nnetsauce::sklearn$model_selection$cross_val_score(obj, X, y)
#
# n <- 25
# p <- 4
#
# set.seed(123)
# X <- matrix(rnorm(n*p), nrow = n, ncol = p)
# y <- sample(c(0L, 1L), n, replace = TRUE)
#
# obj$fit(X, y)
# obj$predict(X)
# obj$score(X, y)
