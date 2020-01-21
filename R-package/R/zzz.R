# global reference to scipy (will be initialized in .onLoad)
# numpy <- NULL
# scipy <- NULL
# sklearn <- NULL
# tqdm <- NULL
ns <- NULL


install_nnetsauce <- function(pip = TRUE) {
  has_nnetsauce <- reticulate::py_module_available("nnetsauce")
  if (has_nnetsauce == FALSE)
      reticulate::py_install("nnetsauce", pip = pip)
}
install_nnetsauce <- memoise::memoise(install_nnetsauce)


.onLoad <- function(libname, pkgname) {
  do.call("install_nnetsauce", list(pip=TRUE))
  # use superassignment to update global reference to numpy
  # numpy <<- reticulate::import("numpy", delay_load = TRUE)
  # scipy <<- reticulate::import("scipy", delay_load = TRUE)
  # sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  # tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
