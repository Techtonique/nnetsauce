# global reference to scipy (will be initialized in .onLoad)
ns <- NULL
sklearn <- NULL

.onLoad <- function(libname, pkgname) {
  # use superassignment to update global reference to package
  ns <<- try(reticulate::import("nnetsauce", delay_load = TRUE),
             silent = TRUE)
  if (inherits(ns, "try-error"))
  {
    reticulate::py_install("nnetsauce")
    ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
  }
  sklearn <<- try(reticulate::import("sklearn", delay_load = TRUE),
                  silent = TRUE)
  if (inherits(sklearn, "try-error"))
  {
    reticulate::py_install("scikit-learn")
    sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  }
}
