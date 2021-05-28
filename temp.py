# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:19:53 2021

@author: holge
"""

import h5py
import numpy as np
import pandas as pd
import seaborn as sns

f = h5py.File('mandelbrot_data', 'r')

data = {'heatmaps': [np.array(f['outputs']['Naive_implementation'][:,:]),
                     np.array(f['outputs']['Numba_implementation'][:,:]),
                     np.array(f['outputs']['Cython_implementation_using_naive_function'][:,:]),
                     np.array(f['outputs']['Cython_implementation_using_vector_function'][:,:]),
                     np.array(f['outputs']['Multiprocessing_implementation_using_vector_function'][:,:]),
                     np.array(f['outputs']['GPU_implementation'][:,:]),
                     np.array(f['outputs']['Distributed_vector_implementation'][:,:])],
        'times': [np.array(f['times']['Naive_implementation']),
                  np.array(f['times']['Numba_implementation']),
                  np.array(f['times']['Cython_implementation_using_naive_function']),
                  np.array(f['times']['Cython_implementation_using_vector_function']),
                  np.array(f['times']['Multiprocessing_implementation_using_vector_function']),
                  np.array(f['times']['GPU_implementation']),
                  np.array(f['times']['Distributed_vector_implementation'])],
        'implementation': ["Naive", "Numba", "Cython_naive", "Cython_vector", 
                           "Multiprocessing", "GPU", "DASK"]
        }

df = pd.DataFrame(data)


# Plot: block diagram implementation vs. execution time
# Plot: heatmaps with mean and name
# 

#%%
# print(f.keys())
# print(f['outputs']['Naive_implementation'][:,:])
# print(np.array(f['times']['Naive_implementation']))

# df.to_hdf('./store.h5', 'data')
# reread = pd.read_hdf('./store.h5')
# # sns.catplot(data = df, x = 'implementation', y = 'times', kind = 'bar')