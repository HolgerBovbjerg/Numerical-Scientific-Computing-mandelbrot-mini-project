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
    range0 = [-2,1.5]
    range1 = [-1.5,1.5]
    res = [400,400]
    
    Re = np.array([np.linspace(-2, 1, res[0]), ] * res[0])
    Im = np.array([np.linspace(-1.5, 1.5, res[1]), ] * res[1]).transpose()
    C = Re + Im * 1j
  
    print("Naive implementation")
    numIter = 1
    start = time.time()
    for i in range(numIter):    
        heatmap = mf.mandelbrot_naive(C,T,I)
        
    naive_time = (time.time() - start)/numIter
    print(f'Execution time:{naive_time} seconds')    
    plt.imshow(heatmap, cmap='hot', interpolation='nearest',extent=[-2, 1, -1.5, 1.5])
    plt.show()

    print("Vectorized implementation")
    numIter = 1
    start = time.time()
    for i in range(numIter):
        heatmap = mf.mandelbrot_vector([C,T,I])

    vector_time = (time.time() - start)/numIter
    print(f'Execution time:{vector_time} seconds')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest',extent=[-2, 1, -1.5, 1.5])
    plt.show()
    
    print("Numba implementation")
    # Run once to compile numba code
    mf.mandelbrot_numba(C,T,I) 
    numIter = 1
    start = time.time()
    for i in range(numIter):
        heatmap = mf.mandelbrot_numba(C,T,I)
    
    numba_time = (time.time() - start)/numIter
    print(f'Execution time:{numba_time} seconds')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])
    plt.show()
    
    print("Cython implementation using naive function")
    numIter = 1
    start = time.time()
    for i in range(numIter):
        heatmap = mf.mandelbrot_naive_cython(C, T, I)

    cython_time = (time.time() - start) / numIter
    print(f'Execution time: {cython_time} seconds')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])
    plt.show()
    
    print("Cython implementation using vector function")
    numIter = 1
    start = time.time()
    for i in range(numIter):
        heatmap = mf.mandelbrot_vector_cython([C, T, I])

    cython_time = (time.time() - start) / numIter
    print(f'Execution time: {cython_time} seconds')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])
    plt.show()
    
    # print("Parallel implementation using vector optimized function")
    # processors = 12
    # numIter = 1
    # start = time.time()
    # for i in range(numIter):
    #     heatmap = mf.mandelbrot_parallel_vector(C, T, I, processors, 12, 2400)

    # parallel_vector_time = (time.time() - start) / numIter
    # print(f'Execution time using {processors} cores: {parallel_vector_time} seconds')
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])
    # plt.show()
    
    # print("Parallel implementation using numba optimized function")
    # processors = 6
    # numIter = 1
    # start = time.time()
    # for i in range(numIter):
    #     heatmap = mf.mandelbrot_parallel_numba(C, T, I, processors, 12, 1200)

    # parallel_numba_time = (time.time() - start) / numIter
    # print(f'Execution time using {processors} cores: {parallel_vector_time} seconds')
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])
    # plt.show()

    # print("GPU implementation ")
    # numIter = 1
    # start = time.time()
    # for i in range(numIter):
    #     heatmap = mf.mandelbrot_GPU(C, T, I)
    # GPU_time = (time.time() - start)/numIter
    # print(f'Execution time:{GPU_time} seconds')

    # plt.imshow(heatmap, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    # plt.title(f'Implementation: GPU, Time: {GPU_time:.2f} seconds')
    # plt.show()
