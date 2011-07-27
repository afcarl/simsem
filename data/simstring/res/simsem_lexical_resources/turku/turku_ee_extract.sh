#!/bin/sh

TB_TUPS=`mktemp`
BASE_FNAME='turku_event_corpus_'

# Collect all text-bound annotations and strip all but type and text
find . -name '*.a1' -o -name '*.a2.t123' \
    | xargs grep -h '^T' \
    | sed -e 's|T[0-9]\+\t\([^ ]\+\) [0-9]\+ [0-9]\+\t\(.*\)|\1 \2|g' \
    > ${TB_TUPS}

# Extract all proteins
grep '^Protein ' ${TB_TUPS} | cut -d ' ' -f 2- | sort | uniq \
    > ${BASE_FNAME}proteins.tokens

# Iterate over the triggers to create the token listings
for TRIGGER_TYPE in 'Binding' 'Entity' 'Gene_expression' 'Localization' \
    'Negative_regulation' 'Phosphorylation' 'Positive_regulation' \
    'Protein_catabolism' 'Regulation' 'Transcription'
do
    FNAME=${BASE_FNAME}triggers.tups.`echo ${TRIGGER_TYPE} \
        | tr '[A-Z]' '[a-z]'`.tokens
    grep "^${TRIGGER_TYPE}" ${TB_TUPS} | cut -d ' ' -f 2- | sort | uniq \
        > ${FNAME}
done

rm -r ${TB_TUPS}
