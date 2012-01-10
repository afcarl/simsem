#!/bin/sh

# Expand all existing corpora annotations to align with NP boundaries.
#
# Note: Requires PTB-style constituency parsers of the text
#
# Author:   Pontus Stenetorp    <pontus stenetorp se>
# Version:  2011-11-07

PARSE_EXT=mcccj

for TXT_FILE in `find data/corpora -name '*.txt'`
do
    ID=`echo ${TXT_FILE} | sed -e 's|\.txt||g' | xargs basename`
    echo "Processing: ${ID}" 1>&2
    DIR=`echo ${TXT_FILE} | xargs dirname` 
    A1_FILE=`find ${DIR} -name "${ID}.*a1"`
    A2_FILE=`find ${DIR} -name "${ID}.*a2"`
    SS_FILE=`find ${DIR} -name "${ID}.*ss"`
    MCCCJ_FILE=`find ${DIR} -name "${ID}.*${PARSE_EXT}"`
    ./tools/npalign.py -n ${TXT_FILE} ${MCCCJ_FILE} ${A1_FILE} ${A2_FILE}
done
