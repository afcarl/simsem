#!/bin/sh

# Prepare for experiments 
#
# Author:   Pontus Stenetorp    <pontus stenetorp se>
# Version:  2011-07-28

(
# Download external resources and make the databases
cd data/simstring/res && gmake ext_res && gmake
) && (
# Build external resources
./build.sh
) && (
# Generate dynamic features from the databases
# NOTE: This is one of the ugliest mistakes in design I have made over the
# last few years, shame on me
classifier/simstring/generate.py features > classifier/simstring/features.py
) && (
# Extract corpora resources
find data/corpora/ -name '*.tar.gz' \
    | xargs -r -I {} sh -c 'tar -x -z -f {} -C `dirname {}`'
) && (
find data/corpora/ -name '*.zip' \
    | xargs -r -I {} sh -c 'unzip -o -d `dirname {}` {}'
)
