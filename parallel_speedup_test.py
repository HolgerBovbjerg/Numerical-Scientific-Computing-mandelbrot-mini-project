import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mandelbrot_functions as mf

if __name__ == "__main__":
    I = 100
    T = 2
    C = mf.create_mesh(2**12, 2**12)
    numIter = 5
    
    # If save_data is true, the plots are saved to pdf-files
    # and the input/output data is saved to a HDF-file
    save_data = True
    
    parallel_vector_time = np.zeros((12,))

    print("Parallel implementation using vector optimized function")
    for j in range(12):
        processors = j + 1
        start = time.time()
        for i in range(numIter):
            heatmap = mf.mandelbrot_parallel_vector(C, T, I, processors, 64, 64)
        parallel_vector_time[j] = (time.time() - start) / numIter
        print(f'Execution time using {processors} cores: {parallel_vector_time[j]} seconds')

    plt.plot([i+1 for i in range(12)], parallel_vector_time)
    plt.title('Time vs. processors')
    plt.ylabel('Average execution time, seconds')
    plt.xlabel('Number of processors')
    plt.savefig('parallel_time_vs_processors.pdf')
    plt.show()

    speedup = [parallel_vector_time[0]/parallel_vector_time[i] for i in range(12)]
    plt.plot([i+1 for i in range(12)], speedup)
    plt.title('Speedup vs processors')
    plt.ylabel('Speedup')
    plt.xlabel('Number of processors')
    plt.savefig('parallel_speedup_vs_processors.pdf')
    plt.show()

    if save_data:
        f = h5py.File('Data/parallel_speedup_data', 'r+')
        input_group = f.create_group('input')
        input_group.create_dataset('complex_input_plane', data=C)
        input_group.create_dataset('threshold_value', data=T)
        input_group.create_dataset('maximum_iterations', data=I)

        time_group = f.create_group('times')
        time_group.create_dataset('core_1', data=parallel_vector_time[0])
        time_group.create_dataset('core_2', data=parallel_vector_time[1])
        time_group.create_dataset('core_3', data=parallel_vector_time[2])
        time_group.create_dataset('core_4', data=parallel_vector_time[3])
        time_group.create_dataset('core_5', data=parallel_vector_time[4])
        time_group.create_dataset('core_6', data=parallel_vector_time[5])
        time_group.create_dataset('core_7', data=parallel_vector_time[6])
        time_group.create_dataset('core_8', data=parallel_vector_time[7])
        time_group.create_dataset('core_9', data=parallel_vector_time[8])
        time_group.create_dataset('core_10', data=parallel_vector_time[9])
        time_group.create_dataset('core_11', data=parallel_vector_time[10])
        time_group.create_dataset('core_12', data=parallel_vector_time[11])
        f.close()
