# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:27:09 2021

@author: holge
"""

import numpy as np

def mandelbrot_naive_cython(c: np.ndarray, T: int, I: int):
    '''
    Function that calculates all M(c) values in the c-mesh given.
    Implemented the naive python way with nested for-loops that calculates
    each M(c) sequentially using the mandelbrot definition.

    :param c: c-mesh containing segment of the complex plane
    :param T: Threshold value used to determine if point is in Mandelbrot set
    :param I: Maximum number of iterations used to determine if point is in Mandelbrot set.
    :return: np.ndarray with M(c) values for each point in c.
    '''
    dim = c.shape
    cdef int x,y
    x = dim[0]
    y = dim[1]
    # cdef int n[x][y] #
    n = np.zeros_like(c, dtype=int)
    for i in range(x):
        for j in range(y):
            z = 0 + 0j
            while abs(z) <= T and n[i, j] < I:
                z = z * z + c[i, j]
                n[i,j] += 1
    return n / I

def mandelbrot_vector_cython(data: list):
    c = data[0]
    cdef float T = data[1]
    cdef int I = data[2]
    z = np.zeros_like(c)
    n = np.zeros_like(c, dtype=int)
    ind = np.full_like(c, True, dtype=bool)
    while np.any(np.abs(z) <= T) and np.all(n < I):
        z[ind] = np.add(np.multiply(z[ind], z[ind]), c[ind])
        ind[np.abs(z) > T] = False
        n[ind] += 1
    return n / I