
# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL


.onLoad <- function(libname, pkgname) {

  foo <- try(reticulate::install_miniconda(),
             silent = FALSE)

  if (class(foo) == "try-error"){
    message("miniconda already installed")
  }

  reticulate::use_python(Sys.which("python"))

  reticulate::use_virtualenv("~/myenv")

  reticulate::conda_create(envname = "myenv")

  #reticulate::use_condaenv("myenv")
  reticulate::conda_install(envname = "myenv",
                            packages = c("numpy", "scipy",
                                         "six", "scikit-learn",
                                         "tqdm", "nnetsauce"),
                            pip = TRUE)

  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
