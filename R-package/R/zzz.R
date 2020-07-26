
# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL
rpy2 <- NULL
# if (.Platform$OS.type == "unix"){
#  jax <- NULL
#  jaxlib <- NULL
# }


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


install_miniconda_ <- function(foo = NULL){

  res <- try(reticulate::install_miniconda(),
             silent = TRUE)
  if (class(res) == "try-error")
  {
    NULL
  } else {
    return(res)
  }
}


install_packages <- function(pip=TRUE) {
  # if (.Platform$OS.type == "unix"){
  #
  #     reticulate::py_install("jax", pip = pip,
  #                            pip_ignore_installed = TRUE)
  #
  #     reticulate::py_install("jaxlib", pip = pip,
  #                            pip_ignore_installed = TRUE)
  # }

    reticulate::py_install("numpy", pip = pip)

    reticulate::py_install("rpy2", pip = pip)

    reticulate::py_install("scipy", pip = pip)

    reticulate::py_install("six", pip = pip)

    reticulate::py_install("tqdm", pip = pip)

   reticulate::py_install("sklearn", pip = pip)

   reticulate::py_install("nnetsauce", pip = pip,
                          pip_ignore_installed = TRUE)
}


.onLoad <- function(libname, pkgname) {

  do.call("uninstall_nnetsauce", list(foo=NULL))

  do.call("install_miniconda_", list(foo=NULL))

  do.call("install_packages", list(pip=TRUE))

  # # use superassignment to update global reference to packages
  # if (.Platform$OS.type == "unix"){
  #   jax <<- reticulate::import("jax", delay_load = TRUE)
  #   jaxlib <<- reticulate::import("jaxlib", delay_load = TRUE)
  # }

  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  rpy2 <<- reticulate::import("rpy2", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
