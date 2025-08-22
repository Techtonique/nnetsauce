import numpy as np
import ctypes
import os

from six import moves
from .sobol import i4_sobol_generate

# From: https://github.com/PhaethonPrime/hammersley/blob/master/hammersley/sequences.py

# this list of primes allows up to a size 120 vector
saved_primes = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
    311,
    313,
    317,
    331,
    337,
    347,
    349,
    353,
    359,
    367,
    373,
    379,
    383,
    389,
    397,
    401,
    409,
    419,
    421,
    431,
    433,
    439,
    443,
    449,
    457,
    461,
    463,
    467,
    479,
    487,
    491,
    499,
    503,
    509,
    521,
    523,
    541,
    547,
    557,
    563,
    569,
    571,
    577,
    587,
    593,
    599,
    601,
    607,
    613,
    617,
    619,
    631,
    641,
    643,
    647,
    653,
    659,
]


def get_phi(p, k):
    p_ = p
    k_ = k
    phi = 0
    while k_ > 0:
        a = k_ % p
        phi += a / p_
        k_ = int(k_ / p)
        p_ *= p
    return phi


# uniform numbers' generation (python)
def generate_uniform(n_dims=2, n_points=10, seed=123):
    np.random.seed(seed=seed)
    return np.random.random((n_dims, n_points))


# hammersley numbers' generation (python)
def generate_hammersley(
    n_dims=2, n_points=100, primes=None, seed=None
):  # seed, just for 'compatibility'
    def func_hammersley(n_dims=n_dims, n_points=(n_points + 1), primes=primes):
        primes = primes if primes is not None else saved_primes
        for k in moves.range(n_points):
            points = [k / n_points] + [
                get_phi(primes[d], k) for d in moves.range(n_dims - 1)
            ]
            yield points

    return np.array(list(func_hammersley()))[1: (n_points + 1), :].transpose()


# halton numbers' generation (python)
def generate_halton(
    n_dims=2, n_points=10, primes=None, seed=None
):  # seed, just for 'compatibility'
    def func_halton(n_dims=n_dims, n_points=(n_points + 1), primes=primes):
        primes = primes if primes is not None else saved_primes
        for k in moves.range(n_points):
            points = [get_phi(primes[d], k) for d in moves.range(n_dims)]
            yield points

    return np.array(list(func_halton()))[1: (n_points + 1), :].transpose()


# sobol numbers' generation (python)
def generate_sobol(
    n_dims=2, n_points=10, seed=None
):  # seed, just for 'compatibility'
    return np.array(i4_sobol_generate(m=n_dims, n=n_points, skip=2))
