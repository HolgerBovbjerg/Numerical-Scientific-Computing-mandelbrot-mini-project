# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:22:41 2021

@author: holge
"""
import time
import matplotlib.pyplot as plt
import mandelbrot_functions as mf

if __name__ == "__main__":
    I = 100
    T = 2
    # C = mf.create_mesh(4096, 4096)
    C = mf.create_mesh(50, 50)
    numIter = 1

    print("Naive implementation")
    start = time.time()
    for i in range(numIter):
        heatmap_naive = mf.mandelbrot_naive(C, T, I)
    naive_time = (time.time() - start) / numIter
    print(f'Execution time:{naive_time:.2f} seconds\n')
    plt.imshow(heatmap_naive, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Naive, Time: {naive_time:.2f} seconds')
    plt.show()

    print("Vectorized implementation")
    data = [C, T, I]
    start = time.time()
    for i in range(numIter):
        heatmap_vector = mf.mandelbrot_vector(data)
    vector_time = (time.time() - start) / numIter
    print(f'Execution time:{vector_time:.2f} seconds\n')
    plt.imshow(heatmap_vector, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Vectorized, Time: {vector_time:.2f} seconds')
    plt.show()

    print("Numba implementation")
    # Run once to compile numba code
    mf.mandelbrot_numba(C, T, I)
    start = time.time()
    for i in range(numIter):
        heatmap_numba = mf.mandelbrot_numba(C, T, I)
    numba_time = (time.time() - start) / numIter
    print(f'Execution time:{numba_time:.2f} seconds\n')
    plt.imshow(heatmap_numba, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: numba, Time: {numba_time:.2f} seconds')
    plt.show()
    # mf.export_figure_matplotlib(heatmap, "20K_Mandelbrot", 146, resize_fact=1, plt_show=False)
    
    print("Cython implementation using naive function")
    numIter = 1
    start = time.time()
    for i in range(numIter):
        heatmap_cython_naive = mf.mandelbrot_naive_cython(C, T, I)

    cython_naive_time = (time.time() - start) / numIter
    print(f'Execution time: {cython_naive_time:.2f} seconds\n')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: cython naive, Time: {cython_naive_time:.2f} seconds')
    plt.show()
    
    print("Cython implementation using vector function")
    numIter = 1
    start = time.time()
    for i in range(numIter):
        heatmap_cython_vector = mf.mandelbrot_vector_cython([C, T, I])

    cython_vector_time = (time.time() - start) / numIter
    print(f'Execution time: {cython_vector_time:.2f} seconds\n')
    plt.imshow(heatmap, cmap='hot', interpolation='nearest', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: cython vector, Time: {cython_vector_time:.2f} seconds')
    plt.show()


    # print("Parallel implementation using vector optimized function")
    # processors = 12
    # start = time.time()
    # for i in range(numIter):
    #     heatmap = mf.mandelbrot_parallel_vector(C, T, I, processors, 512, 8)
    # parallel_vector_time = (time.time() - start) / numIter
    # print(f'Execution time using {processors} cores: {parallel_vector_time:.2f} seconds\n')
    # plt.imshow(heatmap, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    # plt.title(f'Implementation: Parallel vectorized, Time: {parallel_vector_time:.2f} seconds')
    # plt.show()
    
    print('GPU version:')
    start = time.time()
    heatmap_gpu = mf.mandelbrot_gpu(C, T, I)
    GPU_time = (time.time() - start)
    print(f'Execution time:{GPU_time:.2f} seconds\n')
    plt.imshow(heatmap_gpu, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: GPU, Time: {GPU_time:.2f} seconds')
    plt.show()
