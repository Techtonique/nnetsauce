# global reference to scipy (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
sklearn <- NULL
tqdm <- NULL
nnetsauce <- NULL

.onLoad <- function(libname, pkgname) {
  # use superassignment to update global reference to numpy
  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  nnetsauce <<- reticulate::import("nnetsauce", delay_load = TRUE)
}

# nnetsauce::nnetsauce$AdaBoostClassifier
