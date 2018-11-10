from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


ext_modules = [
    Extension('eval_metrics_cy',
              ['eval_metrics_cy.pyx']
    )
]

setup(
    name='Cython-based reid evaluation code',
    ext_modules=cythonize(ext_modules)
)