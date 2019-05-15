#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='robot-brains',
    version='0.1.0',
    author='Bruce Frederiksen',
    author_email='dangyogi@gmail.com',
    description='FTC alternative robot programming language',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dangyogi/robot-brains.git",
    license = "MIT License",
    keywords = "FTC robot programming blockly",
    classifiers = [
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Embedded Systems",
    ],

    #package_dir={'': 'wsgi/erlfrc'},
    packages=['robot_brains'],
    package_data = {
        'robot_brains': ['C_preamble'],
    },
    #py_modules=['manage'],

    entry_points={
        'console_scripts': [
           'mkbrain=robot_brains.parser:compile',
        ],
    },

    install_requires=[
        'ply>=3.11',
    ],
    #dependency_links=[
    #    'https://pypi.python.org/simple/django/'
    #],
)
