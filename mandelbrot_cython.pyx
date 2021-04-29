# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:27:09 2021

@author: holge
"""

import cython
import numpy as np
cimport numpy as np

ctypedef np.complex128_t cpl_t
cpl = np.complex128

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def mandelbrot_naive_cython(np.ndarray[cpl_t,ndim=2] c, int T, int I):
    dim = c.shape
    cdef int x,y
    x = dim[0]
    y = dim[1]
    cdef np.ndarray[int, ndim=2] n = np.zeros((x,y), dtype=int)
    for i in range(x):
        for j in range(y):
            z = 0 + 0j
            while abs(z) <= T and n[i, j] < I:
                z = z * z + c[i, j]
                n[i,j] += 1
    return n / I

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def mandelbrot_vector_cython(data: list):
    dim = data[0].shape
    cdef int x,y
    x = dim[0]
    y = dim[1]
    cdef np.ndarray[cpl_t,ndim=2] c = data[0]
    cdef  int T = data[1]
    cdef int I = data[2]
    cdef np.ndarray[cpl_t, ndim=2] z = np.zeros((x,y),dtype=complex)
    cdef np.ndarray[int, ndim=2] n = np.zeros((x,y), dtype=int)
    cdef np.ndarray[np.uint8_t, ndim = 2, cast=True] ind = np.full((x,y), True, dtype=bool)
    while np.any(np.abs(z) <= T) and np.all(n < I):
        z[ind] = np.add(np.multiply(z[ind], z[ind]), c[ind])
        n[ind] += 1
        ind[np.abs(z) > T] = False
    return n / I