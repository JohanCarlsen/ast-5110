from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np 

extensions = [
    Extension('*', ['*.pyx'],
              include_dirs=[np.get_include()]),
              
]

setup(
    name='Roe flux',
    ext_modules=cythonize(extensions,
                          annotate=True),
    
)