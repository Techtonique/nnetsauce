#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 08:01:48 2019

@author: moudiki
"""

import numpy as np
from ..simulation import generate_sobol2


def optimize_newton(f, x0, max_iter=20, tol=1e-3, h=1e-3):
    def deriv(f, x, h=h):
        return (f(x + h) - f(x - h)) / (2 * h)

    x_prev = x0
    x_next = x_prev - f(x_prev) / deriv(f, x_prev)
    err = np.abs(f(x_next) - f(x_prev))

    num_iter = 0
    while (num_iter <= max_iter) & (err >= tol):
        x_next = x_prev - f(x_prev) / deriv(f, x_prev)
        err = np.abs(f(x_next) - f(x_prev))
        x_prev = x_next
        num_iter += 1

    return (x_next, f(x_next), num_iter, err)


def quasirandom_search(
    func, ubound, lbound, n_points=1000, tol=1e-4
):

    err = np.abs(0)
    d = len(ubound)

    assert d == len(
        lbound
    ), "ubound and lbound must have the same length"

    sobol_seq_choices = np.transpose(
        generate_sobol2(n_dims=d, n_points=n_points)
    )

    x_choices = (
        ubound - lbound
    ) * sobol_seq_choices + lbound

    func_prev = func(x_choices[0, :])
    num_iter = 0
    err = 1e6
    min_x = np.repeat(1000, d)
    min_func = 1e6
    errs = np.zeros(n_points)

    while (err >= tol) & (num_iter < (n_points - 1)):

        x_next = x_choices[(num_iter + 1), :]
        # x_next = (ubound - lbound) * np.random.rand(d) + lbound

        #        print("----------------------------------")
        #        print(num_iter)
        #        print("----------------------------------")
        #        print("\n")
        #
        #        print("x_next")
        #        print(x_next)
        #        print("\n")

        func_next = func(x_next)

        #        print("func_next")
        #        print(func_next)
        #        print("\n")
        #
        #        print("func_prev")
        #        print(func_prev)
        #        print("\n")
        #
        #        print("current min")
        #        print(min_func)
        #        print("\n")

        if func_next < min_func:

            min_x = x_next

            #            print("min_x")
            #            print(min_x)
            #            print("\n")
            #
            #            min_func = func_next
            #
            #            print("min_func")
            #            print(min_func)
            #            print("\n")

            err = np.abs(func_next - func_prev)

        func_prev = func_next

        num_iter += 1

    return (min_x, min_func, errs)


if __name__ == "main":

    from scipy.optimize import rosen, minimize
    import numpy as np
    from os import chdir
    from sklearn.datasets import (
        make_classification,
        load_digits,
        load_breast_cancer,
        load_wine,
        load_iris,
    )
    from sklearn.model_selection import train_test_split
    from time import time
    import matplotlib.pyplot as plt

    wd = (
        "/Users/moudiki/Documents/Python_Packages/nnetsauce"
    )
    chdir(wd)

    import nnetsauce as ns

    #    def objective(x):
    #        return 6*x**5-5*x**4-4*x**3+3*x**2
    #
    #    x_abs = np.linspace(-1, 1.5, num=25)
    #    plt.plot(x_abs, objective(x_abs))

    #    x_vec = np.array([-1, 0, 0.5, 1.5])
    #
    #    [optimize_newton1D(f = objective, x0 = x_vec[i],
    #                    max_iter=20, tol=1e-3) for i in range(len(x_vec))]
    #
    #    # fails (yea)
    #    optimize_newton1D(f = rosen,
    #                     x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2]),
    #                    max_iter=100, tol=1e-6)

    minimize(
        fun=rosen, x0=np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    ).x

    minimize(
        fun=rosen,
        x0=np.array([1.3, 0.7, 0.8, 1.9, 1.2]),
        method="BFGS",
    )

    ns.utils.optim.quasirandom_search(
        func=rosen,
        ubound=np.repeat(-1.5, 5),
        lbound=np.repeat(1.5, 5),
        tol=1e-4,
    )
