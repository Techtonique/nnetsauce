
# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL


install_miniconda_ <- function(silent = TRUE)
{
  try(reticulate::install_miniconda(force=TRUE, update=FALSE),
      silent = silent)
}

uninstall_nnetsauce <- function(foo = NULL) {
  python <- reticulate:::.globals$py_config$python
  packages <- "nnetsauce"
  args <- c("pip", "uninstall", "--yes", packages)
  result <- system2(python, args)
  if (result != 0L) {
    pkglist <- paste(shQuote(packages), collapse = ", ")
    msg <- paste("Error removing package(s):", pkglist)
    stop(msg, call. = FALSE)
  }
  packages
}

install_packages <- function(pip = TRUE) {

  has_numpy <- reticulate::py_module_available("numpy")
  has_scipy <- reticulate::py_module_available("scipy")
  has_six <- reticulate::py_module_available("six")
  has_sklearn <- try(reticulate::py_module_available("scikit-learn"),
                     silent = TRUE)
  if (class(has_sklearn) == "try-error")
  {
    has_sklearn <- reticulate::py_module_available("sklearn")
  }
  has_tqdm <- reticulate::py_module_available("tqdm")

  if (has_numpy == FALSE)
    reticulate::conda_install(packages="numpy", pip = pip)

  if (has_scipy == FALSE)
    reticulate::conda_install(packages="scipy", pip = pip)

  if (has_six == FALSE)
    reticulate::conda_install(packages="six", pip = pip)

  if (has_sklearn == FALSE)
  {
    foo <- try(reticulate::conda_install(packages="scikit-learn", pip = pip),
               silent = TRUE)
    if (class(foo) == "try-error")
    {
      reticulate::conda_install(packages="sklearn", pip = pip)
    }
  }

  if (has_tqdm == FALSE)
    reticulate::conda_install("tqdm", pip = pip)

  foo <- try(reticulate::conda_install(packages="nnetsauce", pip = pip,
                                    pip_ignore_installed = TRUE),
             silent=TRUE)
  if (class(foo) == "try-error")
  {
    reticulate::conda_install(packages="git+https://github.com/Techtonique/nnetsauce.git",
                           pip = pip, pip_ignore_installed = TRUE)
  }
}


.onLoad <- function(libname, pkgname) {

  try(do.call("uninstall_nnetsauce", list(foo=NULL)),
      silent = TRUE)

  do.call("install_miniconda_", list(silent=TRUE))

  do.call("install_packages", list(pip=TRUE))

  # use superassignment to update global reference to packages
  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- try(reticulate::import("sklearn", delay_load = TRUE),
                  silent = TRUE)
  if (class(sklearn) == "try-error")
  {
    sklearn <<- try(reticulate::import("scikit-learn", delay_load = TRUE),
                    silent = TRUE)
  }
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
