import numpy as np
import importlib
import subprocess
import sys
import re

try:
    import pyvinecopulib as pvc

    PYVINECOPULA_AVAILABLE = True
except ImportError:
    PYVINECOPULA_AVAILABLE = False


def vinecopula_sample(x, n_samples=10, method="vine-tll", random_state=123):
    if not PYVINECOPULA_AVAILABLE:
        raise RuntimeError(
            "pyvinecopulib is required for this feature. Install with: pip install nnetsauce[pyvinecopulib] or pip install pyvinecopulib"
        )

    u = pvc.to_pseudo_obs(x)
    method_name = re.sub(r"(?:scp2?-)?vine-", "", method)

    controls = pvc.FitControlsBicop(
        family_set=[getattr(pvc.BicopFamily, method_name)]
    )
    cop = pvc.Vinecop(d=u.shape[1])
    cop.fit(data=u, controls=controls)

    u_sim = cop.simulate(n_samples, seeds=[random_state])
    p = x.shape[1]
    return np.asarray([np.quantile(x[:, i], u_sim[:, i]) for i in range(p)]).T