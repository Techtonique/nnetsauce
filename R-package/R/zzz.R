# global reference to scipy (will be initialized in .onLoad)
ns <- NULL
sklearn <- NULL

.onLoad <- function(libname, pkgname) {
  utils::install.packages("reticulate",
                          repos = list(CRAN = "https://cloud.r-project.org"))
  try(reticulate::virtualenv_create('./r-reticulate'),
      silent = TRUE)
  try(reticulate::use_virtualenv('./r-reticulate'),
      silent = TRUE)
  try(reticulate::py_install(
    "nnetsauce",
    envname = "r-reticulate",
    pip = TRUE,
    pip_options = "--upgrade",
    pip_ignore_installed = TRUE
  ),
  silent = TRUE)
  # use superassignment to update global reference to package
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
}
.onLoad <- memoise::memoise(.onLoad)
