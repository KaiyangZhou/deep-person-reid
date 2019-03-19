from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def find_version():
    version_file = 'torchreid/__init__.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


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
    ]
)