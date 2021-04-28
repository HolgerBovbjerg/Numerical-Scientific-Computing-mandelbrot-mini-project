# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:18:40 2021

@author: holge
"""

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import numba
from numba import jit

def mandelbrot_naive(c, T, I):
    n = np.zeros_like(c, dtype=int)
    dim = c.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            z = 0
            while abs(z) <= T and n[i, j] < I:
                z = z*z + c[i, j]
                n[i, j] += 1
    return n/I


def mandelbrot_vector(c, T, I):
    z = np.zeros_like(c)
    n = np.zeros_like(c, dtype=int)
    ind = np.full_like(c, True, dtype=bool)
    while np.any(np.abs(z) <= T) and np.all(n < I):
        z[ind] = np.add(np.multiply(z[ind], z[ind]), c[ind])
        ind[np.abs(z) > T] = False
        n[ind] += 1
    return n/I


@jit(nopython=True)
def mandelbrot_numba(c, T, I):
    n = np.zeros_like(c, dtype=numba.int64)
    dim = c.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            z = 0
            while abs(z) <= T and n[i, j] < I:
                z = z*z + c[i, j]
                n[i, j] += 1
    return n/I

def mandelbrot_vector_parallel(params):
    c = params[0]
    T = params[1]
    I = params[2]
    z = np.zeros_like(c)
    n = np.zeros_like(c, dtype=int)
    ind = np.full_like(c, True, dtype=bool)
    while np.any(np.abs(z) <= T) and np.all(n < I):
        z[ind] = np.add(np.multiply(z[ind], z[ind]), c[ind])
        ind[np.abs(z) > T] = False
        n[ind] += 1
    return n/I


def mandelbrot_parallel(c, T, I, processors, blockno, blocksize):
    pool = mp.Pool(processes=processors)
    iterable = [tuple((c[blocksize*block:blocksize*block+blocksize],T, I)) for block in range(blockno)]
    results = pool.apply_async(mandelbrot_vector_parallel, iterable)
        
    pool.close()
    pool.join()
    # out_matrix = np.vstack([result.get() for result in results])
    out_matrix = results
    return out_matrix
