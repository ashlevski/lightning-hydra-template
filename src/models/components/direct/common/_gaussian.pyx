# coding=utf-8
# Copyright (c) DIRECT Contributors

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: overflowcheck=False
#cython: unraisable_tracebacks=False

import numpy as np

cimport numpy as cnp
from libc.math cimport cos, log, pi, sin, sqrt
from libc.stdlib cimport RAND_MAX, rand, srand

cnp.import_array()


cdef double random_uniform() nogil:
    """Produces a random number in (0, 1)."""
    cdef double r = rand()
    return r / RAND_MAX


cdef cnp.ndarray[cnp.float_t, ndim=1, mode='c'] random_normal_1d(
    double mu, double std
):
    """Produces a random vector from the Gaussian distribution based on the Box-Muller algorithm."""
    cdef double r, theta, x

    r = sqrt(-2 * log(random_uniform()))
    theta = 2 * pi * random_uniform()

    x = mu + r * cos(theta) * std

    return np.array([x], dtype=float)


cdef cnp.ndarray[cnp.float_t, ndim=1, mode='c'] random_normal_2d(
    double mu_x, double mu_y, double std_x, double std_y
):
    """Produces a random vector from bivariate Gaussian distribution based on the Box-Muller algorithm."""
    cdef double r, theta, x, y

    r = sqrt(-2 * log(random_uniform()))
    theta = 2 * pi * random_uniform()

    x = mu_x + r * cos(theta) * std_x
    y = mu_y + r * sin(theta) * std_y

    return np.array([x, y], dtype=float)


def gaussian_mask_1d(
    int nonzero_count,
    int n,
    int center,
    double std,
    cnp.ndarray[cnp.int_t, ndim=1, mode='c'] mask,
    int seed,
):
    cdef int count, ind
    cdef cnp.ndarray[cnp.float_t, ndim=1, mode='c'] rnd_normal

    srand(seed)

    count = 0

    while count <= nonzero_count:
        rnd_normal = random_normal_1d(center, std)
        ind = int(rnd_normal[0])
        if 0 <= ind < n and mask[ind] != 1:
            mask[ind] = 1
            count = count + 1


def gaussian_mask_2d(
    int nonzero_count,
    int nrow,
    int ncol,
    int center_x,
    int center_y,
    cnp.ndarray[cnp.float_t, ndim=1, mode='c'] std,
    cnp.ndarray[cnp.int_t, ndim=2, mode='c'] mask,
    int seed,
):
    cdef int count, indx, indy
    cdef cnp.ndarray[cnp.float_t, ndim=1, mode='c'] rnd_normal

    srand(seed)

    count = 0

    while count <= nonzero_count:

        rnd_normal = random_normal_2d(center_x, center_y, std[0], std[1])

        indx = int(rnd_normal[0])
        indy = int(rnd_normal[1])

        if 0 <= indx < nrow and 0 <= indy < ncol and mask[indx, indy] != 1:
            mask[indx, indy] = 1
            count = count + 1
