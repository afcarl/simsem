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
classifier/simstring/generate.py features > classifier/simstring/features.py
)
