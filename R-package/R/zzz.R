
# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL


.onLoad <- function(libname, pkgname) {

  reticulate::conda_create(envname = "r-reticulate")

  reticulate::conda_install(envname = "r-reticulate",
                            packages = c("numpy", "scipy",
                                         "six", "scikit-learn",
                                         "tqdm", "nnetsauce"))

  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
