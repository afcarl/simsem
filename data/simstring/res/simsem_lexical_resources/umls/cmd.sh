#!/bin/sh

# Convert UMLS into files with unique tokens.
#
# Author:   Pontus Stenetorp    <pontus stenetorp se>
# Version:  2011-07-27

DATA_FILE=UMLS.ALL

# Convert the UMLS dictionary into separate lists
cat ${DATA_FILE} | ./convert.py

# Unique and sort the lists
find . -maxdepth 1 -name '*.list' | \
    xargs -r -I {} sh -c 'cat {} | sort | uniq > {}.tokens'

# Clean out the lists
rm *.list
