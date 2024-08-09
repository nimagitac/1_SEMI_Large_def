 
from Cython.Build import cythonize
from setuptools import setup
import numpy as np

setup(ext_modules = cythonize("k_geom_cy_optimized_v6.pyx"), include_dirs=[np.get_include()])
