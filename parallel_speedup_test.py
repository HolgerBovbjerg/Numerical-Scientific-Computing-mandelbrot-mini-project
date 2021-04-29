import time
import numpy as np
import matplotlib.pyplot as plt
import mandelbrot_functions as mf

if __name__ == "__main__":
    I = 100
    T = 2
    C = mf.create_mesh(2048, 2048)
    numIter = 10
    parallel_vector_time = np.zeros((12,))

    print("Parallel implementation using vector optimized function")
    for j in range(12):
        processors = j + 1
        start = time.time()
        for i in range(numIter):
            heatmap = mf.mandelbrot_parallel_vector(C, T, I, processors, 512, 4)
        parallel_vector_time[j] = (time.time() - start) / numIter
        print(f'Execution time using {processors} cores: {parallel_vector_time[j]} seconds')

    plt.plot([i+1 for i in range(12)], parallel_vector_time)
    plt.title('Time vs. processors')
    plt.show()

    speedup = [parallel_vector_time[0]/parallel_vector_time[i] for i in range(12)]
    plt.plot([i+1 for i in range(12)], speedup)
    plt.title('Speedup vs processors')
    plt.show()

