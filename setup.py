import os
from distutils.core import setup

VERSION = '0.1.0'


setup(
    name = 'stamps',
    packages = ['stamps'],
    license = 'MIT',
    version = VERSION,
    description = 'Spatial Temporal Analysis and Mapping Python Suite',
    install_requires = [
        'numpy>=1.13', 'scipy>=0.15', 'pandas>=0.22', 'six>=1.11'],
    author = 'stemlab',
    author_email = 'stemlab689@gmail.com',
    url = 'https://github.com/stemlab689/stamps',
    download_url = 'https://github.com/stemlab689/stamps/archive/{v}.tar.gz'\
        .format(v=VERSION),
    keywords = ['analysis', 'mapping', 'stamps'],
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ]
    )
