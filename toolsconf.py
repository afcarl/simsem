'''
Configuration parameters for SimSem.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2010-02-09
'''

# TODO: This really belongs under tools/config.py
# Which should be a package

from os.path import dirname, join
from sys import path

### SimString variables
# Path to the SimString Python Swig module directory
SIMSTRING_LIB_PATH = join(dirname(__file__),
        'external/simstring-1.0/swig/python')

### LibLinear variables
LIBLINEAR_DIR = join(dirname(__file__), 'external/liblinear-1.7')
LIBLINEAR_PYTHON_DIR = join(LIBLINEAR_DIR, 'python')

### Only here until we find a better place to stuff them
LIB_PATH = join(dirname(__file__), 'lib')
