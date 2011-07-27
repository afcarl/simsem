#!/bin/sh

# Extract several types from Entrez Gene data.
#
# Uses extracted tab-separated data from Naoki Okazaki.
#
# Type/fields used:
#
# * Gene locus (6)
# * Protein name (9)
# * Protein description (10)
# * Nomenclature symbol (11)
# * Nomenclature (12)
#
# Author:     Pontus Stenetorp    <pontus stenetorp>
# Version:    2011-07-27

DATA_FILE=entrez_all_names.txt
BASE=entrez_

cut -f 6  ${DATA_FILE} | sed -e 's/|/\n/g' | sed '/^$/d' | \
    sort | uniq > ${BASE}gene_locus.tokens
cut -f 9  ${DATA_FILE} | sed -e 's/|/\n/g' | sed '/^$/d' | \
    sort | uniq > ${BASE}protein_name.tokens
cut -f 10 ${DATA_FILE} | sed -e 's/|/\n/g' | sed '/^$/d' | \
    sort | uniq > ${BASE}protein_description.tokens
cut -f 11 ${DATA_FILE} | sed -e 's/|/\n/g' | sed '/^$/d' | \
    sort | uniq > ${BASE}nomenclature_symbol.tokens
cut -f 12 ${DATA_FILE} | sed -e 's/|/\n/g' | sed '/^$/d' | \
    sort | uniq > ${BASE}nomenclature.tokens
