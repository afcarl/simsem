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
) && (
# Correct offset error in the GREC data set
sed -i -e 's|\(T4\tGene \)423\(.*\)|\1422\2|g' \
    data/corpora/grec/GREC_Standoff/Human/8205615.a1
) && (
# The GREC sentence split files have non-conforming filenames, correct them
find data/corpora/grec -name '*.txt.ss' | sed -e 's|\.txt\.ss||g' \
    | xargs -I {} sh -c 'mv {}.txt.ss {}.ss'
)
