# Numerical-Scientific-Computing-mandelbrot-mini-project
This repository includes code for a mini-project in the Course Numerical Scientific Computing at Aalborg University during spring 2021.
The mini-project deals with implementing algorithms in Python for approximating the Mandelbrot set and mapping each point in complex space to a 2D image illustration of the Mandelbrot set.

## Repository structure
The repository is structured with runfile, function_file, plotfile and test file in the root directory. In addition a GPU kernel is found. 
The ```Cython_files``` folder contains files used in the Cython implementation. 
The ```Data``` folder contains a ```.zip``` file of the generated mandelbrot data.
The ```Info``` folder contains the mini-project description.
The ```Plots``` folder contains plots generated for the mini-project.

## How to use runfile
To generate data ```mandelbrot_runfile.py``` is used. 
The runfile has a number of configuration settings which is:
* Maximum iteration count ```I```
* Threshold ```T``` 
* Size of C-mesh ```C```
* Number of timing runs ```numIter```

## How to generate plots
To generate plots the ```plot_from_file.py``` is used. 
In order to run this file the two data files ```alldata``` and ```mandelbrot_data``` must be in the root folder. 
The data files can be found as ```.zip``` files in the ```Data``` folder.


