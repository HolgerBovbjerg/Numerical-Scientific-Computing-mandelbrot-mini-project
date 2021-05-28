# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:22:41 2021

@author: holge
"""
import time

import numpy as np
import h5py
import matplotlib.pyplot as plt
import mandelbrot_functions as mf

if __name__ == "__main__":
    I = 100
    T = 2
    C = mf.create_mesh(4096, 4096)
    # C = mf.create_mesh(100, 100)
    numIter = 3

    # If save_data is true, the plots are saved to pdf-files
    # and the input/output data is saved to a HDF-file
    save_data = True

    print('GPU implementation')
    GPU_times = []
    for i in range(numIter):
        start = time.time()
        heatmap_gpu = mf.mandelbrot_gpu(C, T, I)
        GPU_times.append(time.time() - start)
    GPU_mean_time = np.mean(GPU_times)
    print(f'Mean execution time:{GPU_mean_time:.2f} seconds\n')
    plt.imshow(heatmap_gpu, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: GPU, Time: {GPU_mean_time:.2f} seconds')
    if save_data:
        mf.export_figure_matplotlib(heatmap_gpu, 'Mandelbrot_gpu.pdf')
    plt.show()


    print("Naive implementation")
    naive_times = []
    for i in range(numIter):
        start = time.time()
        heatmap_naive = mf.mandelbrot_naive(C, T, I)
        naive_times.append((time.time() - start))
    naive_mean_time = np.mean(naive_times)
    print(f'Execution time:{naive_mean_time:.2f} seconds\n')
    plt.imshow(heatmap_naive, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Naive, Time: {naive_mean_time:.2f} seconds')
    if save_data:
        mf.export_figure_matplotlib(heatmap_naive, 'Mandelbrot_naive.pdf')
    plt.show()

    print("Vectorized implementation")
    data = [C, T, I]
    vector_times = []
    for i in range(numIter):
        start = time.time()
        heatmap_vector = mf.mandelbrot_vector(data)
        vector_times.append(time.time() - start)
    vector_mean_time = np.mean(vector_times)
    print(f'Execution time:{vector_mean_time:.2f} seconds\n')
    plt.imshow(heatmap_vector, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Vectorized, Time: {vector_mean_time:.2f} seconds')
    if save_data:
        mf.export_figure_matplotlib(heatmap_vector, 'Mandelbrot_vector.pdf')
    plt.show()
    
    print("Numba implementation")
    # Run once to compile numba code
    mf.mandelbrot_numba(C, T, I)
    numba_times = []
    for i in range(numIter):
        start = time.time()
        heatmap_numba = mf.mandelbrot_numba(C, T, I)
        numba_times.append(time.time() - start)
    numba_mean_time = np.mean(numba_times)
    print(f'Execution time:{numba_mean_time:.2f} seconds\n')
    plt.imshow(heatmap_numba, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Numba, Time: {numba_mean_time:.2f} seconds')
    if save_data:
        mf.export_figure_matplotlib(heatmap_numba, 'Mandelbrot_numba.pdf')
    plt.show()

    print("Cython implementation using naive function")
    cython_naive_times = []
    for i in range(numIter):
        start = time.time()
        heatmap_cython_naive = mf.mandelbrot_naive_cython(C, T, I)
        cython_naive_times.append(time.time() - start) 
    cython_naive_mean_time = np.mean(cython_naive_times)
    print(f'Execution time: {cython_naive_mean_time:.2f} seconds\n')
    plt.imshow(heatmap_cython_naive, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Cython naive, Time: {cython_naive_mean_time:.2f} seconds')
    if save_data:
        mf.export_figure_matplotlib(heatmap_cython_naive, 'Mandelbrot_cython_naive.pdf')
    plt.show()

    print("Cython implementation using vector function")
    cython_vector_times = []
    for i in range(numIter):
        start = time.time()
        heatmap_cython_vector = mf.mandelbrot_vector_cython([C, T, I])
        cython_vector_times.append(time.time() - start) 
    cython_vector_mean_time = np.mean(cython_vector_times)
    print(f'Execution time: {cython_vector_mean_time:.2f} seconds\n')
    plt.imshow(heatmap_cython_naive, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Cython vector, Time: {cython_vector_mean_time:.2f} seconds')
    if save_data:
        mf.export_figure_matplotlib(heatmap_cython_naive, 'Mandelbrot_cython_vector.pdf')
    plt.show()

    print("Multiprocessing implementation using vector function")
    processors = 12
    parallel_vector_times = []
    for i in range(numIter):
        start = time.time()
        # heatmap_parallel = mf.mandelbrot_parallel_vector(C, T, I, processors, 512, 8)
        heatmap_parallel = mf.mandelbrot_parallel_vector(C, T, I, processors, 20, 5)
        parallel_vector_times.append(time.time() - start)
    parallel_vector_mean_time = np.mean(parallel_vector_times)
    print(f'Execution time using {processors} cores: {parallel_vector_mean_time:.2f} seconds\n')
    plt.imshow(heatmap_parallel, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Parallel vectorized, Time: {parallel_vector_mean_time:.2f} seconds')
    if save_data:
        mf.export_figure_matplotlib(heatmap_parallel, 'Mandelbrot_parallel.pdf')
    plt.show()


    print('Distributed vector implementation')
    processors = 12
    distributed_vector_times = []
    for i in range(numIter):
        start = time.time()
        # heatmap_dist_vec = mf.mandelbrot_distribu1ted_vector(C, T, I, processors, 512, 8)
        heatmap_dist_vec = mf.mandelbrot_distributed_vector(C, T, I, processors, 20, 5)
        distributed_vector_times.append(time.time() - start)
    distributed_vector_mean_time = np.mean(distributed_vector_times)
    print(f'Execution time using {processors} cores: {distributed_vector_mean_time:.2f} seconds\n')
    plt.imshow(heatmap_dist_vec, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Distributed vectorized, Time: {distributed_vector_mean_time:.2f} seconds')
    if save_data:
        mf.export_figure_matplotlib(heatmap_dist_vec, 'Mandelbrot_distributed.pdf')
    plt.show()


    if save_data:
        f = h5py.File('mandelbrot_data', 'w-')
        input_group = f.create_group('input')
        input_group.create_dataset('complex_input_plane', data=C)
        input_group.create_dataset('threshold_value', data=T)
        input_group.create_dataset('maximum_iterations', data=I)

        output_group = f.create_group('outputs')
        output_group.create_dataset('Naive_implementation', data=heatmap_naive)
        output_group.create_dataset('Vectorized_implementation', data=heatmap_vector)
        output_group.create_dataset('Numba_implementation', data=heatmap_numba)
        output_group.create_dataset('Cython_implementation_using_naive_function', data=heatmap_cython_naive)
        output_group.create_dataset('Cython_implementation_using_vector_function', data=heatmap_cython_naive)
        output_group.create_dataset('Multiprocessing_implementation_using_vector_function', data=heatmap_parallel)
        output_group.create_dataset('GPU_implementation', data=heatmap_gpu)
        output_group.create_dataset('Distributed_vector_implementation', data=heatmap_dist_vec)

        time_group = f.create_group('times')
        time_group.create_dataset('Naive_implementation', data=naive_times)
        time_group.create_dataset('Vectorized_implementation', data=vector_times)
        time_group.create_dataset('Numba_implementation', data=numba_times)
        time_group.create_dataset('Cython_implementation_using_naive_function', data=cython_naive_times)
        time_group.create_dataset('Cython_implementation_using_vector_function', data=cython_vector_times)
        time_group.create_dataset('Multiprocessing_implementation_using_vector_function', data=parallel_vector_times)
        time_group.create_dataset('GPU_implementation', data=GPU_times)
        time_group.create_dataset('Distributed_vector_implementation', data=distributed_vector_times)
        f.close()
