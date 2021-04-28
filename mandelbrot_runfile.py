# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:22:41 2021

@author: holge
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import mandelbrot_functions as mf

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

if __name__ == "__main__":
    I = 100
    T = 10
    range0 = [-2,1.5]
    range1 = [-1.5,1.5]
    res = [500,500]
    
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
    
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest',extent=[-2, 1, -1.5, 1.5])
    
    # print("Vectorized implementation")
    # numIter = 10
    # start = time.time()
    # for i in range(numIter):
    #     heatmap = mf.mandelbrot_vector(C,T,I)
        
    # vector_time = (time.time() - start)/numIter
    # print(f'Execution time:{vector_time} seconds')
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest',extent=[-2, 1, -1.5, 1.5])
    
    # print("Numba implementation")
    # # Run once to compile numba code
    # mf.mandelbrot_numba(C,T,I) 
    # numIter = 10
    # start = time.time()
    # for i in range(numIter):
    #     heatmap = mf.mandelbrot_numba(C,T,I)
    
    # numba_time = (time.time() - start)/numIter
    # print(f'Execution time:{numba_time} seconds')
    
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest',extent=[-2, 1, -1.5, 1.5])
              

    print("Parallel implementation using vector optimized function")
    processors = 12 
    numIter = 1
    start = time.time()
    for i in range(numIter):
        heatmap = mandelbrot_parallel(C,T,I, processors, 25, 500)
    
    parallel_vector_time = (time.time() - start)/numIter
    print(f'Execution time using {processors} cores: {parallel_vector_time} seconds')
    
    # plt.imshow(heatmap, cmap='hot', interpolation='nearest',extent=[-2, 1, -1.5, 1.5])
              