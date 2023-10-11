#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: overflowcheck=False
#cython: unraisable_tracebacks=False

import numpy as np
cimport numpy as cnp
from libc.math cimport cos, pi, sin
from libc.stdlib cimport RAND_MAX, rand, srand

cnp.import_array()


cdef double random_uniform() nogil:
    """Produces a random number in (0, 1)."""
    cdef double r = rand()
    return r / RAND_MAX


cdef int randint(int upper) nogil:
    """Produces a random integer in {0, 1, ..., upper-1}."""
    return int(random_uniform() * (upper))


cdef inline Py_ssize_t fmax(Py_ssize_t one, Py_ssize_t two) nogil:
    """Max(a, b)."""
    return one if one > two else two


cdef inline Py_ssize_t fmin(Py_ssize_t one, Py_ssize_t two) nogil:
    """Min(a, b)."""
    return one if one < two else two


def poisson(
    int nx,
    int ny,
    int max_attempts,
    cnp.ndarray[cnp.int_t, ndim=2, mode='c'] mask,
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] radius_x,
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] radius_y,
    int seed
):
    """
    Notes
    -----

    * Code inspired and modified from [1]_ with BSD-3 licence, Copyright (c) 2016, Frank Ong, Copyright (c) 2016,
        The Regents of the University of California [2]_.

    References
    ----------

    .. [1] https://github.com/mikgroup/sigpy/blob/1817ff849d34d7cbbbcb503a1b310e7d8f95c242/sigpy/mri/samp.py#L158
    .. [2] https://github.com/mikgroup/sigpy/blob/master/LICENSE
    """

    cdef int x, y, num_actives, i, k
    cdef float rx, ry, v, t, qx, qy, distance
    cdef Py_ssize_t startx, endx, starty, endy, px, py

    # initialize active list
    cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] pxs = np.empty(nx * ny, dtype=int)
    cdef cnp.ndarray[cnp.int_t, ndim=1, mode='c'] pys = np.empty(nx * ny, dtype=int)

    srand(seed)

    with nogil:

        pxs[0] = randint(nx)
        pys[0] = randint(ny)

        num_actives = 1

        while num_actives > 0:
            # Select a sample from active list
            i = randint(num_actives)
            px = pxs[i]
            py = pys[i]
            rx = radius_x[px, py]
            ry = radius_y[px, py]

            # Attempt to generate point
            done = False
            k = 0

            while not done and k < max_attempts:

                # Generate point randomly from r and 2 * r
                v = random_uniform() + 1
                t = 2 * pi * random_uniform()
                qx = px + v * rx * cos(t)
                qy = py + v * ry * sin(t)

                # Reject if outside grid or close to other points
                if qx >= 0 and qx < nx and qy >= 0 and qy < ny:
                    startx = fmax(int(qx - rx), 0)
                    endx = fmin(int(qx + rx + 1), nx)
                    starty = fmax(int(qy - ry), 0)
                    endy = fmin(int(qy + ry + 1), ny)

                    done = True
                    for x in range(startx, endx):
                        for y in range(starty, endy):
                            distance = ((qx - x) / radius_x[x, y]) ** 2 + ((qy - y) / (radius_y[x, y])) ** 2
                            if (mask[x, y] == 1) and (distance < 1):
                                done = False
                                break

                k += 1

            # Add point if done else remove from active list
            if done:
                pxs[num_actives] = int(qx)
                pys[num_actives] = int(qy)
                mask[pxs[num_actives], pys[num_actives]] = 1
                num_actives += 1
            else:
                num_actives -= 1
                pxs[i] = pxs[num_actives]
                pys[i] = pys[num_actives]
