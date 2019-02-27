from setuptools import setup, find_packages
from distutils.core import Extension

dataloader = Extension(
        'OpenKE.libdataloader',
        sources=['base/Base.cpp'])

setup(
        name='OpenKE',
        version='0.0.1',
        packages=find_packages(),
        install_requires=[
            'tensorflow',
            'numpy',
            ],
        ext_modules=[dataloader])
