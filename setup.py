from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


def readme():
    with open('README.rst') as f:
        content = f.read()
    return content


def find_version():
    version_file = 'torchreid/__init__.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        'torchreid.metrics.rank_cylib.rank_cy',
        ['torchreid/metrics/rank_cylib/rank_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]


setup(
    name='torchreid',
    version=find_version(),
    description='Pytorch framework for deep-learning person re-identification',
    author='Kaiyang Zhou',
    author_email='k.zhou.vision@gmail.com',
    license='MIT',
    long_description=readme(),
    url='https://github.com/KaiyangZhou/deep-person-reid',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'Cython',
        'h5py',
        'Pillow',
        'six',
        'scipy>=1.0.0',
        'torch>=0.4.1',
        'torchvision>=0.2.1'
    ],
    keywords=[
        'Person Re-Identification',
        'Deep Learning',
        'Computer Vision'
    ],
    ext_modules=cythonize(ext_modules)
)