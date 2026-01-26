from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        'torchreid.metrics.rank_cylib.rank_cy',
        ['torchreid/metrics/rank_cylib/rank_cy.pyx'],
        include_dirs=[np.get_include()],
    )
]

setup(ext_modules=cythonize(ext_modules))
