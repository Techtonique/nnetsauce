import numpy as np
import importlib
import subprocess
import sys


def install_package(package_name):
    """Install a package dynamically using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def check_and_install(package_name):
    """Check if a package is installed; if not, install it."""
    try:
        # Check if the package is already installed by importing it
        importlib.import_module(package_name)        
    except ImportError:
        print(f"'{package_name}' not found. Installing...")
        install_package(package_name)
        # Retry importing the package after installation
        importlib.import_module(package_name)
        print(f"'{package_name}' has been installed successfully.")

def vinecopula_sample(x, n_samples=10, method="vine-tll", random_state=123):
    check_and_install(pyvinecopulib)
    u = pyvinecopulib.to_pseudo_obs(x)
    method_name = method.replace("scp-vine-", "")
    method_name = method_name.replace("scp2-vine-", "")
    method_name = method_name.replace("vine-", "")    
    controls = pyvinecopulib.FitControlsVinecop(
        family_set=[getattr(pyvinecopulib.BicopFamily, method_name)]
    )
    cop = pyvinecopulib.Vinecop(u, controls=controls)
    u_sim = cop.simulate(n_samples, seeds=[random_state])
    p = x.shape[1]
    return np.asarray([np.quantile(x[:, i], u_sim[:, i]) for i in range(p)]).T
