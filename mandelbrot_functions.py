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
import pyopencl as cl


def mandelbrot_naive(c, T, I):
    n = np.zeros_like(c, dtype=int)
    dim = c.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            z = 0
            while abs(z) <= T and n[i, j] < I:
                z = z * z + c[i, j]
                n[i, j] += 1
    return n / I


def mandelbrot_vector(c, T, I):
    z = np.zeros_like(c)
    n = np.zeros_like(c, dtype=int)
    ind = np.full_like(c, True, dtype=bool)
    while np.any(np.abs(z) <= T) and np.all(n < I):
        z[ind] = np.add(np.multiply(z[ind], z[ind]), c[ind])
        ind[np.abs(z) > T] = False
        n[ind] += 1
    return n / I


@jit(nopython=True)
def mandelbrot_numba(c, T, I):
    n = np.zeros_like(c, dtype=numba.int64)
    dim = c.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            z = 0
            while abs(z) <= T and n[i, j] < I:
                z = z * z + c[i, j]
                n[i, j] += 1
    return n / I


def mandelbrot_parallel_vector(c, T, I, processors, blockno, blocksize):
    pool = mp.Pool(processes=processors)
    results = pool.map_async(mandelbrot_vector, [tuple(
        (c[blocksize * block:blocksize * block + blocksize],
         T,
         I)
    ) for block in range(blockno)]
                             )
    pool.close()
    pool.join()
    # out_matrix = np.vstack([result.get() for result in results])
    out_matrix = results.get()
    return out_matrix


def mandelbrot_GPU(c, T, I):
    result_matrix = np.empty_like(c).astype(np.float64)

    # Set up the GPU stuff
    # Create a context and choose device interactively
    ctx = cl.create_some_context(interactive=True)

    # If interactive mode does not work, use this snippet instead.
    # platform = cl.get_platforms()[0]
    # my_device = platform.get_devices()[0]
    # ctx = cl.Context([my_device])

    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    c_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, result_matrix.nbytes)

    kernelsource = open("mandelbrot_kernel.cl").read()
    prg = cl.Program(ctx, kernelsource).build()

    # Execute the "mandelbrot" kernel in the program.
    mandelbrot = prg.mandelbrot
    mandelbrot.set_scalar_arg_dtypes([None, None, np.int32, np.int32])
    mandelbrot(
        queue,  # Command queue
        (50, 50),  # Global grid size
        None,  # Work group size
        c_gpu,  # param 0
        result_g,  # param 1
        I,  # param 2
        T,  # param 3
    )
    cl.enqueue_copy(queue, result_matrix, result_g)

    return result_matrix
