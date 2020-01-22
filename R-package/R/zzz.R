# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL


install_miniconda_ <- function(silent = TRUE)
{
  try(reticulate::install_miniconda(), silent = TRUE)
}


install_packages <- function(pip = TRUE) {

  has_numpy <- reticulate::py_module_available("numpy")
  has_scipy <- reticulate::py_module_available("scipy")
  has_sklearn <- reticulate::py_module_available("sklearn")
  has_tqdm <- reticulate::py_module_available("tqdm")
  has_nnetsauce <- reticulate::py_module_available("nnetsauce")

  if (has_numpy == FALSE)
    reticulate::py_install("numpy", pip = pip)

  if (has_scipy == FALSE)
    reticulate::py_install("scipy", pip = pip)

  if (has_sklearn == FALSE)
    reticulate::py_install("sklearn", pip = pip)

  if (has_tqdm == FALSE)
    reticulate::py_install("tqdm", pip = pip)

  if (has_nnetsauce == FALSE)
    reticulate::py_install("nnetsauce", pip = pip)
}


.onLoad <- function(libname, pkgname) {

  # has_reticulate <- "reticulate" %in% rownames(utils::installed.packages())
  # if (has_reticulate == FALSE)
  #  utils::install.packages("reticulate")

  do.call("install_miniconda_", list(silent=TRUE))

  do.call("install_packages", list(pip=TRUE))

  # use superassignment to update global reference to packages
  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)

}
