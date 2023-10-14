# global reference to scipy (will be initialized in .onLoad)
ns <- NULL
sklearn <- NULL

.onLoad <- function(libname, pkgname) {
  utils::install.packages("reticulate",
                          repos = list(CRAN="https://cloud.r-project.org"))

  try(reticulate::virtualenv_create('./r-reticulate'),
      silent = TRUE)
  try(reticulate::use_virtualenv('./r-reticulate'),
      silent = TRUE)
  # reticulate::py_install("scikit-learn",
  #                        pip = TRUE,
  #                        pip_ignore_installed = TRUE)
  reticulate::py_install("nnetsauce",
                         pip = TRUE,
                         pip_ignore_installed = TRUE)
  # try(reticulate::py_install("scikit-learn",
  #                        envname = "r-reticulate",
  #                        pip = TRUE,
  #                        pip_ignore_installed = TRUE),
  #     silent = TRUE)
  try(reticulate::py_install("nnetsauce",
                         envname = "r-reticulate",
                         pip = TRUE,
                         pip_ignore_installed = TRUE),
      silent = TRUE)
  # use superassignment to update global reference to package
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
}
