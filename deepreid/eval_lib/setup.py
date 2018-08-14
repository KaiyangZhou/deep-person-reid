import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()
print(numpy_include)

ext_modules = [Extension("cython_eval",
                         ["eval.pyx"],
                         libraries=["m"],
                         include_dirs=[numpy_include],
                         extra_compile_args=["-ffast-math", "-Wno-cpp", "-Wno-unused-function"]
                         ),
               ]

setup(
    name='eval_lib',
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules)
