from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="isfactor", ext_modules=cythonize("isfactor.pyx")
)
