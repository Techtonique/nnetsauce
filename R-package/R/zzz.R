# global reference to packages (will be initialized in .onLoad)
numpy <- NULL
scipy <- NULL
six <- NULL
sklearn <- NULL
tqdm <- NULL
ns <- NULL
rp <- NULL
if (as.character(Sys.info()[1]) %in% c("Linux", "Darwin")){
 rjax <- NULL
 rjaxlib <- NULL
}


install_miniconda_ <- function(silent = TRUE)
{
  try(reticulate::install_miniconda(), 
      silent = TRUE) # loop in requirements.txt instead
}

install_packages <- function(pip = TRUE) {

  if (as.character(Sys.info()[1]) %in% c("Linux", "Darwin")){
    has_jax <- reticulate::py_module_available("jax")
    has_jaxlib <- reticulate::py_module_available("jaxlib")
  }  
  has_nnetsauce <- reticulate::py_module_available("nnetsauce")
  has_numpy <- reticulate::py_module_available("numpy")
  has_rpy2 <- reticulate::py_module_available("rpy2")
  has_scipy <- reticulate::py_module_available("scipy")
  has_sklearn <- reticulate::py_module_available("sklearn")
  has_six <- reticulate::py_module_available("six")
  has_tqdm <- reticulate::py_module_available("tqdm")
  has_nnetsauce <- reticulate::py_module_available("nnetsauce")

  if (as.character(Sys.info()[1]) %in% c("Linux", "Darwin")){

    if (has_jax == FALSE)
      reticulate::py_install("jax", pip = pip)

    if (has_jaxlib == FALSE)
      reticulate::py_install("jaxlib", pip = pip)  

  }

  if (has_numpy == FALSE)
    reticulate::py_install("numpy", pip = pip)

  if (has_rpy2 == FALSE)
    reticulate::py_install("rpy2", pip = pip)

  if (has_scipy == FALSE)
    reticulate::py_install("scipy", pip = pip)

  if (has_six == FALSE)
    reticulate::py_install("six", pip = pip)  

  if (has_sklearn == FALSE)
    reticulate::py_install("sklearn", pip = pip)

  if (has_tqdm == FALSE)
    reticulate::py_install("tqdm", pip = pip)

  if (has_nnetsauce == FALSE)
    reticulate::py_install("nnetsauce", pip = pip)    
}


.onLoad <- function(libname, pkgname) {

  do.call("install_miniconda_", list(silent=TRUE))

  do.call("install_packages", list(pip=TRUE))

  # use superassignment to update global reference to packages
  if (as.character(Sys.info()[1]) %in% c("Linux", "Darwin")){
  rjax <<- reticulate::import("jax", delay_load = TRUE)
  rjaxlib <<- reticulate::import("jaxlib", delay_load = TRUE)
  }
  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  six <<- reticulate::import("six", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)  
}
