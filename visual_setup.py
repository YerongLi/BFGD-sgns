from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Word Embedding',
    ext_modules = cythonize(['util/visualize.pyx'])

)
