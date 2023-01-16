
# global reference to packages (will be initialized in .onLoad)
joblib <- NULL
numpy <- NULL
pandas <- NULL
scipy <- NULL
sklearn <- NULL
threadpoolctl <- NULL
tqdm <- NULL
ns <- NULL

install_packages <- function(pip=TRUE) {

    reticulate::py_install("joblib", pip = pip)
    reticulate::py_install("numpy", pip = pip)
    reticulate::py_install("pandas", pip = pip)
    reticulate::py_install("scipy", pip = pip)
    reticulate::py_install("scikit-learn", pip = pip)
    reticulate::py_install("threadpoolctl", pip = pip)
    reticulate::py_install("tqdm", pip = pip)

   foo <- try(reticulate::py_install("nnetsauce", pip = pip,
                                      pip_ignore_installed = TRUE),
               silent=TRUE)
    if (class(foo) == "try-error")
    {
     reticulate::py_install("git+https://github.com/Techtonique/nnetsauce.git",
                            pip = pip, pip_ignore_installed = TRUE)
   }

}


.onLoad <- function(libname, pkgname) {

  do.call("install_packages", list(pip=TRUE))

  joblib <<- reticulate::import("joblib", delay_load = TRUE)
  numpy <<- reticulate::import("numpy", delay_load = TRUE)
  pandas <<- reticulate::import("pandas", delay_load = TRUE)
  scipy <<- reticulate::import("scipy", delay_load = TRUE)
  sklearn <<- reticulate::import("sklearn", delay_load = TRUE)
  threadpoolctl <<- reticulate::import("threadpoolctl", delay_load = TRUE)
  tqdm <<- reticulate::import("tqdm", delay_load = TRUE)
  ns <<- reticulate::import("nnetsauce", delay_load = TRUE)
}
