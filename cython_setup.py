# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:28:51 2021

@author: holge
"""

from setuptools import setup
from Cython.Build import cythonize

import numpy



setup(
    ext_modules = cythonize("mandelbrot_cython.pyx"),
    include_dirs=[numpy.get_include()]
)