# Authors: T. Moudiki
#
# License: BSD 3 Clause Clear

import numpy as np
from tqdm import tqdm
from ..sampling.helpers import dosubsample


# 0 - utils -----

# 0 - 0 data structures & funcs -----


def call_f(f, x):
    return f(x)


def calc_grad(f, x):
    return numerical_gradient(f, x)


def calc_hessian(f, x):
    return numerical_hessian(f, x)


def generate_index(response, batch_prop=1.0, randomization="strat", seed=123):
    """Generation of indices for Stochastic gradient descent."""

    n = len(response)

    if batch_prop < 1:
        if randomization == "strat":
            return dosubsample(response, batch_prop, seed=seed)

        if randomization == "shuffle":
            np.random.seed(seed)
            return np.asarray(
                np.floor(
                    np.random.choice(
                        range(n), size=np.floor(batch_prop * n), replace=False
                    )
                ),
                dtype=np.int,
            )
    return None


def check_is_min(f, x):
    return (numerical_hessian(f, x)[1]) * 1


def calc_fplus(x, alpha, p_k):
    p = len(x)
    res = np.zeros(p)
    j = 0

    for j in range(p):
        res[j] = x[j] + alpha * p_k[j]

    return res


# Useful functions -----

# 1 - gradient-----


def numerical_gradient(f, x, **kwargs):
    p = len(x)
    ix = 0
    eps_factor = 6.055454452393343e-06  # machine eps ** (1/3)
    zero = 2.220446049250313e-16
    res = np.zeros_like(x)

    def f_(x):
        return f(x, **kwargs)

    for ix in range(p):
        value_x = x[ix]

        h = max(eps_factor * value_x, 1e-8)

        x[ix] = value_x + h
        fx_plus = call_f(f_, x)

        x[ix] = value_x - h
        fx_minus = call_f(f_, x)

        x[ix] = value_x  # restore

        res[ix] = (fx_plus - fx_minus) / (2 * h)

    return np.asarray(res)


# 2 - hessian-----


def numerical_hessian(f, x, **kwargs):
    p = len(x)
    ix = 0
    jx = 0
    eps_factor = 0.0001220703125  # machine eps ** (1/4)
    zero = 2.220446049250313e-16
    H = np.zeros((p, p))
    temp = 0

    def f_(x):
        return f(x, **kwargs)

    fx = call_f(f_, x)

    for ix in range(p):
        for jx in range(ix, p):
            if ix < jx:
                value_x = x[ix]
                value_y = x[jx]

                h = max(eps_factor * value_x, 1e-8)
                k = max(eps_factor * value_y, 1e-8)

                x[ix] = value_x + h
                x[jx] = value_y + k
                fx_plus = call_f(f_, x)

                x[ix] = value_x + h
                x[jx] = value_y - k
                fx_plus_minus = call_f(f_, x)

                x[ix] = value_x - h
                x[jx] = value_y + k
                fx_minus_plus = call_f(f_, x)

                x[ix] = value_x - h
                x[jx] = value_y - k
                fx_minus = call_f(f_, x)

                x[ix] = value_x  # restore
                x[jx] = value_y  # restore

                temp = (fx_plus - fx_plus_minus - fx_minus_plus + fx_minus) / (
                    4 * h * k
                )

            else:
                value_x = x[ix]

                h = max(eps_factor * value_x, 1e-8)

                x[ix] = value_x + h
                fx_plus = call_f(f_, x)

                x[ix] = value_x - h
                fx_minus = call_f(f_, x)

                x[ix] = value_x  # restore

                temp = (fx_plus - 2 * fx + fx_minus) / (h**2)

            H[ix, jx] = temp
            H[jx, ix] = temp

    res = np.asarray(H)

    return res, all(np.linalg.eig(res)[0] > 0)


# 3 - One-hot encoder -----


def one_hot_encode(y, n_classes):
    n_obs = len(y)
    res = np.zeros((n_obs, n_classes), dtype=float)

    for i in range(n_obs):
        res[i, y[i]] = 1

    return np.asarray(res)


# Coordinate descent (Stochastic) -----

# 1 - algos -----


def scd(
    loss_func,
    response,
    x,
    num_iters=200,
    batch_prop=1.0,
    learning_rate=0.01,
    mass=0.9,
    decay=0.1,
    method="momentum",
    randomization="strat",
    tolerance=1e-3,
    verbose=1,
    **kwargs,
):
    """Stochastic coordinate descent with momentum and adaptive learning rates."""

    i = 0
    j = 0
    n = len(response)
    p = len(x)
    velocity = np.zeros(p)
    losses = []

    if verbose == 1:
        iterator = tqdm(range(num_iters))
    else:
        iterator = range(num_iters)

    def f(x):
        return loss_func(x, **kwargs)

    if method == "momentum":
        for i in iterator:
            idx = generate_index(
                response=response,
                batch_prop=batch_prop,
                randomization=randomization,
                seed=i,
            )

            def f_j(h, xx, j):
                value_x = 0
                res = 0
                value_x = xx[j]
                xx[j] = xx[j] + h
                res = loss_func(xx, row_index=idx, **kwargs)
                xx[j] = value_x
                return res

            diff = -np.asarray(x)

            if verbose == 2:
                print(f"\n x prev: {np.asarray(x)}")

            for j in range(p):
                h0 = 6.055454452393343e-06 * x[j]
                grad_x = (f_j(h0, x, j) - f_j(-h0, x, j)) / (2 * h0)
                velocity[j] = mass * velocity[j] - learning_rate * grad_x
                x[j] = x[j] + velocity[j]

            diff += np.asarray(x)

            if verbose == 2:
                print(f"\n x new: {np.asarray(x)}")

            losses.append(f(x))

            if (len(losses) > 3) and (
                np.abs(np.diff(losses[-2:])[0]) < tolerance
            ):
                break

            if verbose == 2:
                print("\n")
                print(f"iter {i+1} - decrease -----")

                try:
                    print(np.linalg.norm(diff, 1))
                except:
                    pass

                print(f"iter {i+1} - loss -----")
                print(np.flip(losses)[0])

    if method in ("exp", "poly"):
        for i in iterator:
            idx = generate_index(
                response=response,
                batch_prop=batch_prop,
                randomization=randomization,
                seed=i,
            )

            def f_j(h, xx, j):
                value_x = 0
                res = 0
                value_x = xx[j]
                xx[j] = xx[j] + h
                res = loss_func(xx, row_index=idx, **kwargs)
                xx[j] = value_x
                return res

            decay_rate = (
                (1 + decay * i) if (method == "poly") else np.exp(decay * i)
            )

            diff = -np.asarray(x)

            if verbose == 2:
                print(f"\n x prev: {np.asarray(x)}")

            losses.append(f(x))

            for j in range(p):
                h0 = 6.055454452393343e-06 * x[j]
                grad_x = (f_j(h0, x, j) - f_j(-h0, x, j)) / (2 * h0)
                x[j] = x[j] - grad_x * learning_rate / decay_rate

            diff += np.asarray(x)

            if verbose == 2:
                print(f"\n x new: {np.asarray(x)}")

            if (len(losses) > 3) and (
                np.abs(np.diff(losses[-2:])[0]) < tolerance
            ):
                break

            if verbose == 2:
                print("\n")
                print(f"iter {i+1} - decrease -----")

                try:
                    print(np.linalg.norm(diff, 1))
                except:
                    pass

                print(f"iter {i+1} - loss -----")
                print(np.flip(losses)[0])

    return np.asarray(x), num_iters, losses


# Gradient descent (Stochastic) -----


def sgd(
    loss_func,
    response,
    x,
    num_iters=200,
    batch_prop=1.0,
    learning_rate=0.01,
    mass=0.9,
    decay=0.1,
    method="momentum",
    randomization="strat",
    tolerance=1e-3,
    verbose=1,
    **kwargs,
):
    """Stochastic gradient descent with momentum and adaptive learning rates."""

    i = 0
    j = 0
    n = len(response)
    p = len(x)
    velocity = np.zeros(p)
    grad_i = np.zeros(p)
    losses = []

    if verbose == 1:
        iterator = tqdm(range(num_iters))
    else:
        iterator = range(num_iters)

    def f(x):
        return loss_func(x, **kwargs)

    if method == "momentum":
        for i in iterator:
            idx = generate_index(
                response=response,
                batch_prop=batch_prop,
                randomization=randomization,
                seed=i,
            )

            def objective(x):
                return loss_func(x, row_index=idx, **kwargs)

            # grad_i = numerical_gradient(objective, x)
            grad_i = calc_grad(objective, x)

            diff = -np.asarray(x)

            if verbose == 2:
                print(f"\n x prev: {np.asarray(x)}")

            for j in range(p):
                velocity[j] = mass * velocity[j] - learning_rate * grad_i[j]
                x[j] = x[j] + velocity[j]

            diff += np.asarray(x)

            if verbose == 2:
                print(f"\n x new: {np.asarray(x)}")

            losses.append(f(x))

            if (len(losses) > 3) and (
                np.abs(np.diff(losses[-2:])[0]) < tolerance
            ):
                break

            if verbose == 2:
                print("\n")
                print(f"iter {i+1} - decrease -----")

                try:
                    print(np.linalg.norm(diff, 1))
                except:
                    pass

                print(f"iter {i+1} - loss -----")
                print(np.flip(losses)[0])

    if method in ("exp", "poly"):
        for i in iterator:
            idx = generate_index(
                response=response,
                batch_prop=batch_prop,
                randomization=randomization,
                seed=i,
            )

            def objective(x):
                return loss_func(x, row_index=idx, **kwargs)

            # grad_i = numerical_gradient(objective, x)
            grad_i = calc_grad(objective, x)

            decay_rate = (
                (1 + decay * i) if (method == "poly") else np.exp(decay * i)
            )

            diff = -np.asarray(x)

            if verbose == 2:
                print(f"\n x prev: {np.asarray(x)}")

            for j in range(p):
                x[j] = x[j] - grad_i[j] * learning_rate / decay_rate

            if verbose == 2:
                print(f"\n x new: {np.asarray(x)}")

            diff += np.asarray(x)

            losses.append(f(x))

            if (len(losses) > 3) and (
                np.abs(np.diff(losses[-2:])[0]) < tolerance
            ):
                break

            if verbose == 2:
                print("\n")
                print(f"iter {i+1} - decrease -----")

                try:
                    print(np.linalg.norm(diff, 1))
                except:
                    pass

                print(f"iter {i+1} - loss -----")
                print(np.flip(losses)[0])

    return np.asarray(x), num_iters, losses
