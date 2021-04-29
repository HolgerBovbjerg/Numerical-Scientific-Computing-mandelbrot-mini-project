# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:18:40 2021

@author: holge
"""

import numpy as np
import multiprocessing as mp
import numba
from numba import jit
import matplotlib.pyplot as plt
import pyopencl as cl
import mandelbrot_cython as mc
from dask.distributed import Client, wait


def mandelbrot_naive(c: np.ndarray, T: int, I: int):
    """
    Function that calculates all M(c) values in the c-mesh given.
    Implemented the naive python way with nested for-loops that calculates
    each M(c) sequentially using the mandelbrot definition.

    :param c: c-mesh containing segment of the complex plane
    :param T: Threshold value used to determine if point is in Mandelbrot set
    :param I: Maximum number of iterations used to determine if point is in Mandelbrot set.
    :return: np.ndarray with M(c) values for each point in c.
    """
    n = np.zeros_like(c, dtype=int)
    dim = c.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            z = 0
            while abs(z) <= T and n[i, j] < I:
                z = z * z + c[i, j]
                n[i, j] += 1
    return n / I

def mandelbrot_vector(data: list):
    """
    Function that calculates the M(c) values in the c-mesh given,
    implemented in a vectorised way using numpy.
    Here each point in the mesh is updated "at once" at each iteration.
    In order to use this function for the multiprocessing and distributed functions
    the input has been packed into a list.

    :param data: Data is a list containing:
        c: c-mesh containing segment of the complex plane
        T: Threshold value used to determine if point is in Mandelbrot set
        I: Maximum number of iterations used to determine if point is in Mandelbrot set.
    :return: np.ndarray with M(c) values for each point in c.
    """
    c = data[0]
    T = data[1]
    I = data[2]
    z = np.zeros_like(c)
    n = np.zeros_like(c, dtype=int)
    ind = np.full_like(c, True, dtype=bool)
    while np.any(np.abs(z) <= T) and np.all(n < I):
        n[ind] += 1
        z[ind] = np.add(np.multiply(z[ind], z[ind]), c[ind])
        ind[np.abs(z) >= T] = False

    return np.real(n) / I


@jit(nopython=True)
def mandelbrot_numba(c: np.ndarray, T: int, I: int):
    """
    Function that calculates the M(c) values in the c-mesh given,
    implemented using the numba library on the naive implementation.
    Numba analyzes and optimizes the code before compiling it to a
    machine code version tailored to the CPU.
    For more info on Numba, visit https://numba.pydata.org/

    :param c: c-mesh containing segment of the complex plane
    :param T: Threshold value used to determine if point is in Mandelbrot set
    :param I: Maximum number of iterations used to determine if point is in Mandelbrot set.
    :return: np.ndarray with M(c) values for each point in c.
    """
    n = np.zeros_like(c, dtype=numba.int64)
    dim = c.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            z = 0
            while abs(z) <= T and n[i, j] < I:
                z = z * z + c[i, j]
                n[i, j] += 1
    return n / I


def mandelbrot_parallel_vector(c: np.ndarray, T: int, I: int, processors: int, blockno: int, blocksize: int):
    """
    Function that calculates the M(c) values in the c-mesh given,
    using multiprocessing to do calculate in parallel.
    The functions uses the python multiprocessing library and assigns work
    with the asynchronous map function.
    The c-mesh is divided into equal size blocks, and each block is sent
    to the vectorized mandelbrot function.
    A block consists of one or more whole rows of c-mesh.

    :param c: c-mesh containing segment of the complex plane
    :param T: Threshold value used to determine if point is in Mandelbrot set
    :param I: Maximum number of iterations used to determine if point is in Mandelbrot set.
    :param processors: Number of processors to divide the workload amongst.
    :param blockno: Number of blocks to divide the c-mesh into.
    :param blocksize: The amount of rows of c-mesh in a single block.
    :return: np.ndarray with M(c) values for each point in c-mesh.
    """
    pool = mp.Pool(processes=processors)
    data = [[c[blocksize * block:blocksize * block + blocksize], T, I] for block in range(blockno)]
    results = pool.map_async(mandelbrot_vector, data)

    pool.close()
    pool.join()
    out_matrix = np.vstack([row for row in results.get()])
    return out_matrix


def mandelbrot_gpu(c: np.ndarray, T: int, I: int):
    """
    Function that calculates the M(c) values in the c-mesh given,
    using GPU-processing.
    This function uses the PyOpenCL library for python.
    The calculations are performed by applying a kernel written
    in C++.
    This function uses the kernel "mandelbrot_kernel.cl" located in
    the parent folder.
    The kernel is designed to use private memory when possible, to
    reduce the overhead from calling global memory.

    :param c: c-mesh containing segment of the complex plane
    :param T: Threshold value used to determine if point is in Mandelbrot set
    :param I: Maximum number of iterations used to determine if point is in Mandelbrot set.
    :return: np.ndarray with M(c) values for each point in c-mesh.
    """
    # Initialize a matrix to contain the results.
    result_matrix = np.empty(c.shape).astype(np.float64)

    # Set up the GPU stuff by choosing a platform and a device.
    # Depending on the machine, the indices might need to be changed
    # to get the actual GPU device.
    platform = cl.get_platforms()[0]
    my_device = platform.get_devices()[0]

    ctx = cl.Context([my_device])
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    c_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)
    result_g = cl.Buffer(ctx, mf.WRITE_ONLY, result_matrix.nbytes)

    # Find the kernel from the kernel-file and build the program.
    kernelsource = open("mandelbrot_kernel.cl").read()
    prg = cl.Program(ctx, kernelsource).build()

    # Execute the "mandelbrot" kernel in the program.
    mandelbrot = prg.mandelbrot
    mandelbrot.set_scalar_arg_dtypes([None, None, np.int32, np.int32])
    mandelbrot(
        queue,  # Command queue
        c.shape,  # Global grid size
        None,  # Work group size
        c_gpu,  # param 0
        result_g,  # param 1
        I,  # param 2
        T,  # param 3
    )
    cl.enqueue_copy(queue, result_matrix, result_g)

    return result_matrix


def mandelbrot_naive_cython(C, T, I):
    """
    Function that calculates all M(c) values in the c-mesh given.
    Implemented the naive python way with nested for-loops that calculates
    each M(c) sequentially using the mandelbrot definition. The function is
    then optimized using cython (see mandelbrot_cython.pyx).

    :param c: c-mesh containing segment of the complex plane
    :param T: Threshold value used to determine if point is in Mandelbrot set
    :param I: Maximum number of iterations used to determine if point is in Mandelbrot set.
    :return: np.ndarray with M(c) values for each point in c.
    """
    return mc.mandelbrot_naive_cython(C, T, I)
 
    
def mandelbrot_vector_cython(data: list):
    """
    Function that calculates the M(c) values in the c-mesh given,
    implemented in a vectorised way using numpy. This is then optmized using
    cython (see mandelbrot_cython.pyx).
    Here each point in the mesh is updated "at once" at each iteration.
    In order to use this function for the multiprocessing and distributed functions
    the input has been packed into a list. The function is
    then optimized using cython (see mandelbrot_cython.pyx).

    :param data: Data is a list containing:
        c: c-mesh containing segment of the complex plane
        T: Threshold value used to determine if point is in Mandelbrot set
        I: Maximum number of iterations used to determine if point is in Mandelbrot set.
    :return: np.ndarray with M(c) values for each point in c.
    """
    return mc.mandelbrot_vector_cython(data)


def mandelbrot_distributed(c: np.ndarray, T: int, I: int, processes: int, blockno: int, blocksize: int):
    client = Client(n_workers=processes)

    data = [[c[blocksize * block:blocksize * block + blocksize], T, I] for block in range(blockno)]
    counts = client.map(mandelbrot_vector, data)
    client.gather(counts)
    rows = np.vstack([count.result() for count in counts])

    wait(rows)
    client.close()

    return rows


def distributed_vector_mandel(processes, c_mesh, max_iterations, threshold_value):
    client = Client(n_workers=processes)
    counts = []

    for i in range(16):
        # Send the data to the cluster as this is best practice for large data.
        big_future = client.scatter(c_mesh[250 * i:250 * i + 250])
        counts.append(
            client.submit(mandelbrot_vector, big_future, max_iterations, threshold_value)
        )

    client.gather(counts)
    rows = np.vstack([count.result()[0] for count in counts])
    wait(rows)

    client.close()

    return rows


def export_figure_matplotlib(arr, f_name, dpi=200, resize_fact=1, plt_show=False):
    """
    Export array as figure in original resolution
    :param arr: array of image to save in original resolution
    :param f_name: name of file where to save figure
    :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
    :param dpi: dpi of your screen
    :param plt_show: show plot or not
    """
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1] / dpi, arr.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr)
    plt.savefig(f_name, dpi=(dpi * resize_fact))
    if plt_show:
        plt.show()
    else:
        plt.close()


def create_mesh(real_points: int, imag_points: int):
    """
    Function that generates a mesh of complex points from the complex
    plane, in the region: -2 < Re < 1  and -1.5 < Im < 1.5
    The resolution of the mesh is determined by the input values.
    :param real_points: Number of points on the real axis
    :param imag_points: Number of points on the imaginary axis
    :return: 2D ndarray of complex values.
    """
    Re = np.array([np.linspace(-2, 1, real_points), ] * real_points)
    Im = np.array([np.linspace(-1.5, 1.5, imag_points), ] * imag_points).transpose()
    return Re + Im * 1j
