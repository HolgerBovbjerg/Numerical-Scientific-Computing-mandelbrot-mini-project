# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:22:41 2021

@author: holge
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import mandelbrot_functions as mf

if __name__ == "__main__":
    I = 100
    T = 10
    range0 = [-2, 1.5]
    range1 = [-1.5, 1.5]
    res = [50, 50]

    Re = np.array([np.linspace(-2, 1, res[0]), ] * res[0])
    Im = np.array([np.linspace(-1.5, 1.5, res[1]), ] * res[1]).transpose()
    C = Re + Im * 1j

    print("Naive implementation")
    numIter = 10
    start = time.time()
    for i in range(numIter):
        heatmap = mf.mandelbrot_naive(C, T, I)

    naive_time = (time.time() - start) / numIter
    print(f'Execution time:{naive_time} seconds')

    plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])

    print("Vectorized implementation")
    numIter = 10
    data = [C, T, I]
    start = time.time()
    for i in range(numIter):
        heatmap = mf.mandelbrot_vector(data)

    vector_time = (time.time() - start) / numIter
    print(f'Execution time:{vector_time} seconds')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])

    print("Numba implementation")
    # Run once to compile numba code
    mf.mandelbrot_numba(C, T, I)
    numIter = 10
    start = time.time()
    for i in range(numIter):
        heatmap = mf.mandelbrot_numba(C, T, I)

    numba_time = (time.time() - start) / numIter
    print(f'Execution time:{numba_time} seconds')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])

    print("Parallel implementation using vector optimized function")
    processors = 6
    numIter = 10
    start = time.time()
    for i in range(numIter):
        heatmap = mf.mandelbrot_parallel_vector(C, T, I, processors, 5, 50)

    parallel_vector_time = (time.time() - start) / numIter
    print(f'Execution time using {processors} cores: {parallel_vector_time} seconds')
    plt.imshow(heatmap[0], cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])
    plt.show()

    print('GPU version:')
    start = time.time()
    result_matrix = mf.mandelbrot_GPU(C, T, I)
    GPU_time = (time.time() - start)
    print(f'Execution time:{GPU_time} seconds')

    plt.imshow(result_matrix, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: GPU, Time: {GPU_time:.2f} seconds')
    plt.show()
