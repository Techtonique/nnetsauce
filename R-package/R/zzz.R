
# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL

# https://rstudio.github.io/reticulate/articles/package.html#installing-python-dependencies
install_packages <- function(conda = "auto") {

  reticulate::conda_install(envname = "r-reticulate",
                            packages = c("numpy", "scipy", "six",
                                         "tqdm", "scikit-learn",
                                         "nnetsauce"),
                            conda = conda)
}


.onLoad <- function(libname, pkgname) {

  foo <- try(expr = reticulate::use_condaenv("r-reticulate"),
             silent = FALSE)
  if (class(foo) == "try-error"){
    foo <- reticulate::conda_create(envname = "r-reticulate",
                                    packages = c("numpy", "scipy", "six",
                                                 "tqdm", "scikit-learn",
                                                 "nnetsauce"))
  }

  do.call(what = "install_packages",
          args = list(conda = "auto"))

  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
