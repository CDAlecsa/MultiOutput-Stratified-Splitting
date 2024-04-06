# Load modules
import os
import numpy

from distutils.core import setup, Extension
from Cython.Build import cythonize

extensions = [Extension("_genetic_algorithm", ["_genetic_algorithm.pyx"]), 
              Extension("_utils", ["_utils.pyx"])
            ]


# Cython compilation: use "python setup.py build_ext --inplace"
setup(
    ext_modules = cythonize(extensions, compiler_directives = {"language_level": "3"}, annotate = True),
    package_dir = {os.path.dirname(__file__): ''},
    include_dirs = [numpy.get_include()],
)

