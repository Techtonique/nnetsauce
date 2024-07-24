import numpy as np
import pyvinecopulib as pv


def vinecopula_sample(x, n_samples=10, method="vine-tll", random_state=123):
    u = pv.to_pseudo_obs(x)
    method_name = method.replace("scp-vine-", "")
    method_name = method_name.replace("scp2-vine-", "")
    method_name = method_name.replace("vine-", "")
    controls = pv.FitControlsVinecop(
        family_set=[getattr(pv.BicopFamily, method_name)]
    )
    cop = pv.Vinecop(u, controls=controls)
    u_sim = cop.simulate(n_samples, seeds=[random_state])
    p = x.shape[1]
    return np.asarray([np.quantile(x[:, i], u_sim[:, i]) for i in range(p)]).T
