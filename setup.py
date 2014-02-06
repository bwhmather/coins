from setuptools import setup, find_packages
from setuptools.dist import Distribution
from setuptools.extension import Extension


## Packages required for build (pulled in by Cython modules)
Distribution(dict(setup_requires=['cython', 'numpy']))

from Cython.Build import cythonize
import numpy as np


setup(
    name='Coins',
    version='0.0.1',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'matplotlib',

        'tornado',
        'jinja2',
        'pyzmq',
        'pillow',
        'ipython',
        ],
    tests_require=[
        'nose',
        'pep8',
        'pyflakes',
        ],
    setup_requires=[
        'numpy',
        'cython',
        ],
    packages=find_packages(),
    ext_modules=cythonize([
        Extension('coins._hough', ['coins/_hough.pyx'],
                  include_dirs=[np.get_include()]
            ),
        ]),
    )
