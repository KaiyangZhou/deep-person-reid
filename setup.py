import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


ext_modules = [
    Extension(
        'torchreid.metrics.rank_cylib.rank_cy',
        ['torchreid/metrics/rank_cylib/rank_cy.pyx'],
        include_dirs=[np.get_include()],
    )
]


setup(
    packages=find_packages(),
    ext_modules=cythonize(ext_modules)
)

