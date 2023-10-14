# global reference to scipy (will be initialized in .onLoad)
ns <- NULL
sklearn <- NULL

.onLoad <- function(libname, pkgname) {
  utils::install.packages("reticulate")
  utils::update.packages("reticulate")
  reticulate::py_install("nnetsauce",
                         pip = TRUE,
                         pip_ignore_installed = TRUE)
  # use superassignment to update global reference to package
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
}
