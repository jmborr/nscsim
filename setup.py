#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import nsc

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'future', 'numpy', 'six', 'tqdm'
    # TODO: put package requirements here
]

setup_requirements = [
    'pytest-runner',
    # TODO(jmborr): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

setup(
    name='nsc',
    version=nsc.__version__,
    long_description=readme,
    author="Jose Borreguero",
    author_email='borreguero@gmail.com',
    url='https://github.com/jmborr/nsc',
    packages=find_packages(include=['nsc']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='nsc',
    classifiers=[
        'Development Status :: 0 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
