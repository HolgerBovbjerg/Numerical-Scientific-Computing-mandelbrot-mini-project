# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:28:51 2021

@author: holge
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("mandelbrot_cython.pyx"),
)