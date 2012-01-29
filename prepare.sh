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
# Build external resources
./build.sh
) && (
# Download external resources and make the databases
cd data/simstring/res && make_cmd ext_res && make_cmd
) && (
# Generate dynamic features from the databases
# NOTE: This is one of the ugliest mistakes in design I have made over the
# last few years, shame on me
classifier/simstring/generate.py features > classifier/simstring/features.py
)
