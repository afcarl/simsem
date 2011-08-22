#!/bin/sh

# Prepare for experiments 
#
# Author:   Pontus Stenetorp    <pontus stenetorp se>
# Version:  2011-07-28

# Apparently some GNU/Linux distributions won't even supply `gmake`, so we are
# forced to do this qualified guess which will now and then clash when we
# leave the world of GNU...
make_cmd() {
    # XXX: Ugly hack, but will work for now, `hash` won't play along
    gmake ${@} || make ${@}
}

(
# Download external resources and make the databases
cd data/simstring/res && make_cmd ext_res && make_cmd
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
