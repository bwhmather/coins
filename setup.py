from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

setup(
  name = 'Coins',
  ext_modules = cythonize([
      Extension('coins._hough', ['coins/_hough.pyx'],
                include_dirs=[np.get_include()]
      ),
  ]),
)
