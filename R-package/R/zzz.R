# global reference to scipy (will be initialized in .onLoad)
ns <- NULL
sklearn <- NULL

.onLoad <- function(libname, pkgname) {
  utils::install.packages("reticulate",
                          repos = list(CRAN="http://cran.rstudio.com/"))
  utils::update.packages("reticulate")
  reticulate::virtualenv_create("r-reticulate")
  reticulate::use_virtualenv("r-reticulate")
  reticulate::py_install("scikit-learn",
                         envname = "r-reticulate",
                         pip = TRUE,
                         pip_ignore_installed = TRUE)
  reticulate::py_install("nnetsauce",
                         envname = "r-reticulate",
                         pip = TRUE,
                         pip_ignore_installed = TRUE)
  # use superassignment to update global reference to package
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
}
