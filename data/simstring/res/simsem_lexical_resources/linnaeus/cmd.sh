#!/bin/sh

cat dict-species.tsv \
    | cut -f 2 \
    | sed 's/|/\n/g' \
    | sort \
    | uniq \
    > dicts-species.tsv.tokens
