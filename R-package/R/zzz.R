
# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL


.onLoad <- function(libname, pkgname) {

  foo <- try(reticulate::use_python(python = Sys.which("python3"),
                         required = TRUE), silent = FALSE)
  if (class(foo) == "try-error")
  {
    message("Skipping use_python")
  }

  foo2 <- try(reticulate::virtualenv_create("r-reticulate",
                                python = Sys.which("python3")),
              silent = FALSE)
  if (class(foo2) == "try-error")
  {
    message("Skipping virtualenv_create")
  }

  foo3 <- try(reticulate::use_condaenv(condaenv = "r-reticulate"),
              silent = FALSE)
  if (class(foo3) == "try-error")
  {
    message("Skipping use_condaenv")
  }

  foo4 <- try(reticulate::conda_install(envname = "r-reticulate",
                            packages = c("numpy", "scipy",
                                         "six", "scikit-learn",
                                         "tqdm", "nnetsauce"),
                            pip = TRUE), silent = FALSE)
  if (class(foo4) == "try-error")
  {
    message("Skipping conda_install")
  }

  foo5 <- try(reticulate::use_condaenv("r-reticulate"),
              silent = FALSE)
  if (class(foo5) == "try-error")
  {
    message("Skipping use_condaenv")
  }

  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
