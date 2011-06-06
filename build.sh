#!/bin/sh

# Quick hack to extract and compile all external dependencies.
# Author:   Pontus Stenetorp    <pontus stenetorp se>
# Version:  2011-06-07

ARCH_DIR='external'

LIBLINEAR='liblinear-1.7'
( cd ${ARCH_DIR} && rm -rf ${LIBLINEAR} && tar xfz ${LIBLINEAR}.tar.gz && \
    cd ${LIBLINEAR} && make && cd python && make )

SIMSTRING='simstring-1.0'
( cd ${ARCH_DIR} && rm -rf ${SIMSTRING} && tar xfz ${SIMSTRING}.tar.gz && \
    cd ${LIBLINEAR} && ./configure && cd swig/python && ./prepare.sh && \
    python setup.py build_ext --inplace )

LIBSVM='libsvm-3.1'
( cd ${ARCH_DIR} && rm -rf ${LIBSVG} && tar xfz ${LIBSVM}.tar.gz && \
    cd ${LIBSVM} && make && cd python && make )
