from setuptools import setup, find_packages
from distutils.core import Extension

dataloader = Extension(
        'OpenKE.libdataloader',
        sources=['base/Base.cpp'])

setup(
        name='OpenKE',
        version='0.0.1',
        packages=find_packages(),
        extra_packages = {
            'tensorflow': ['tensorflow>=1.13.1'],
            'tensorflow with gpu': ['tensorflow-gpu>=1.13.1']
            },
        install_requires=[
            'numpy',
            ],
        ext_modules=[dataloader],
)
