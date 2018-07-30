from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'Word Embedding',
    ext_modules = cythonize(['util/word2vec_as_MF.pyx','util/functions.pyx'])

)
