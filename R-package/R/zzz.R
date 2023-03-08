
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
    message("Skipping use_python...")
  }

  foo1 <- try(reticulate::install_miniconda(force = TRUE), silent = FALSE)
  if (class(foo1) == "try-error")
  {
    message("Skipping install_miniconda...")
  }

  foo2 <- try(reticulate::virtualenv_create("r-reticulate",
                                python = Sys.which("python3"),
                                packages = c("numpy", "scipy", "six",
                                             "scikit-learn", "tqdm",
                                             "nnetsauce")),
              silent = FALSE)
  if (class(foo2) == "try-error")
  {
    message("Skipping virtualenv_create...")
  }

  foo3 <- try(reticulate::use_condaenv(condaenv = "r-reticulate"),
              silent = FALSE)
  if (class(foo3) == "try-error")
  {
    message("Skipping use_condaenv...")
  }

  foo5 <- try(reticulate::use_condaenv("r-reticulate"),
              silent = FALSE)
  if (class(foo5) == "try-error")
  {
    message("Skipping use_condaenv...")
  }

  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
