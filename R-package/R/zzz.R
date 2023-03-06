
# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL


.onLoad <- function(libname, pkgname) {

  foo <- try(reticulate::install_miniconda(force = FALSE),
             silent=FALSE)
  if (class(foo) == "try-error")
  {
    message("Not reinstalling miniconda...")
  }

  foo2 <- try(reticulate::conda_create(envname = "r-reticulate",
                            packages = c("numpy", "scipy", "six",
                                         "tqdm", "scikit-learn",
                                         "nnetsauce")), silent = FALSE)
  if (class(foo2) == "try-error")
  {
    message("Not re-creating r-reticulate ...")
  }

  foo3 <- try(reticulate::use_condaenv(condaenv = "r-reticulate"),
              silent = FALSE)
  if (class(foo3) == "try-error")
  {
    message("Using r-reticulate as codaenv failed...")
  }

  foo4 <- try(reticulate::conda_install(envname = "r-reticulate",
                            packages = c("numpy", "scipy", "six",
                                        "tqdm", "scikit-learn",
                                          "nnetsauce")),
              silent = FALSE)
  if (class(foo4) == "try-error")
  {
    message("Using conda_install for installing packages failed...")
  }


  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
