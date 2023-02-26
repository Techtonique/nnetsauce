
# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL


install_miniconda_ <- function(silent = TRUE)
{
  try(reticulate::install_miniconda(),
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

install_packages <- function(method = "auto", conda = "auto") {

  reticulate::py_install(packages = c("numpy", "scipy", "six",
                                      "tqdm", "scikit-learn",
                                      "nnetsauce"), pip = TRUE,
  method = method, conda = conda)

}


.onLoad <- function(libname, pkgname) {

  #try(do.call("uninstall_nnetsauce", list(foo=NULL)),
  #    silent = TRUE)

  # do.call("install_miniconda_", list(silent=TRUE))

  do.call("install_packages", list(method = "auto", conda = "auto"))

  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
