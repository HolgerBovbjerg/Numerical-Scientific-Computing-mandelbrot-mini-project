# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:22:41 2021

@author: holge
"""
import time
import matplotlib.pyplot as plt
import mandelbrot_functions as mf
import h5py

if __name__ == "__main__":
    I = 100
    T = 2
    C = mf.create_mesh(4096, 4096)
    C = mf.create_mesh(100, 100)
    numIter = 1

    # If save_data is true, the plots are saved to pdf-files
    # and the input/output data is saved to a HDF-file
    save_data = True

    print("Naive implementation")
    start = time.time()
    for i in range(numIter):
        heatmap_naive = mf.mandelbrot_naive(C, T, I)
    naive_time = (time.time() - start) / numIter
    print(f'Execution time:{naive_time:.2f} seconds\n')
    plt.imshow(heatmap_naive, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Naive, Time: {naive_time:.2f} seconds')
    if save_data:
        plt.savefig('Mandelbrot_naive.pdf')
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
    if save_data:
        plt.savefig('Mandelbrot_vector.pdf')
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
    plt.title(f'Implementation: Numba, Time: {numba_time:.2f} seconds')
    if save_data:
        plt.savefig('Mandelbrot_numba.pdf')
    plt.show()

    print("Cython implementation using naive function")
    start = time.time()
    for i in range(numIter):
        heatmap_cython_naive = mf.mandelbrot_naive_cython(C, T, I)
    cython_naive_time = (time.time() - start) / numIter
    print(f'Execution time: {cython_naive_time:.2f} seconds\n')
    plt.imshow(heatmap_cython_naive, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Cython naive, Time: {cython_naive_time:.2f} seconds')
    if save_data:
        plt.savefig('Mandelbrot_cython_naive.pdf')
    plt.show()

    print("Cython implementation using vector function")
    start = time.time()
    for i in range(numIter):
        heatmap_cython_vector = mf.mandelbrot_vector_cython([C, T, I])

    cython_vector_time = (time.time() - start) / numIter
    print(f'Execution time: {cython_vector_time:.2f} seconds\n')
    plt.imshow(heatmap_cython_naive, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Cython vector, Time: {cython_vector_time:.2f} seconds')
    if save_data:
        plt.savefig('Mandelbrot_cython_vector.pdf')
    plt.show()

    print("Parallel implementation using vector optimized function")
    processors = 12
    start = time.time()
    for i in range(numIter):
        # heatmap_parallel = mf.mandelbrot_parallel_vector(C, T, I, processors, 512, 8)
        heatmap_parallel = mf.mandelbrot_parallel_vector(C, T, I, processors, 20, 5)
    parallel_vector_time = (time.time() - start) / numIter
    print(f'Execution time using {processors} cores: {parallel_vector_time:.2f} seconds\n')
    plt.imshow(heatmap_parallel, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Parallel vectorized, Time: {parallel_vector_time:.2f} seconds')
    if save_data:
        plt.savefig('Mandelbrot_parallel.pdf')
    plt.show()

    print('GPU implementation')
    start = time.time()
    heatmap_gpu = mf.mandelbrot_gpu(C, T, I)
    GPU_time = (time.time() - start)
    print(f'Execution time:{GPU_time:.2f} seconds\n')
    plt.imshow(heatmap_gpu, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: GPU, Time: {GPU_time:.2f} seconds')
    if save_data:
        plt.savefig('Mandelbrot_gpu.pdf')
    plt.show()

    print('Distributed vector implementation')
    processors = 12
    start = time.time()
    for i in range(numIter):
        # heatmap_dist_vec = mf.mandelbrot_distributed_vector(C, T, I, processors, 512, 8)
        heatmap_dist_vec = mf.mandelbrot_distributed_vector(C, T, I, processors, 20, 5)
    distributed_vector_time = (time.time() - start) / numIter
    print(f'Execution time using {processors} cores: {distributed_vector_time:.2f} seconds\n')
    plt.imshow(heatmap_dist_vec, cmap='hot', extent=[-2, 1, -1.5, 1.5])
    plt.title(f'Implementation: Parallel vectorized, Time: {distributed_vector_time:.2f} seconds')
    if save_data:
        plt.savefig('Mandelbrot_distributed.pdf')
    plt.show()

    if save_data:
        f = h5py.File('Mandelbrot_datasets', 'w-')
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
        output_group.create_dataset('Parallel implementation using vector optimized function', data=heatmap_parallel)
        output_group.create_dataset('GPU_implementation', data=heatmap_gpu)
        output_group.create_dataset('Distributed_vector_implementation', data=heatmap_dist_vec)
        f.close()
