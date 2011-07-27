#!/usr/bin/env python

'''
Expand GO-terms to new variants.

Author:     Pontus Stenetorp
Version:    2009-07-17
Version:    2011-04-04
'''

import re
import sys

### Cons
# The following rules are all derived from the paper:
#       "Lexical Properties of OBO Ontology Class Names and Synonyms"
#               By: Elena Beisswanger, Michael Poprat and Udo Hahn
# Can generate 2^8 new variants at most
PAPER_RULES = (
        # Replace with space
        (re.compile(r'_'), ' '),
        (re.compile(r'-'), ' '),

        # Remove
        # TODO: Read the paper, how are their rules? Try merging? Effect?
        (re.compile(r'[\[\]]'), ''),
        (re.compile(r'[\(\)]'), ''),
        (re.compile(r"'"), ''),

        # Comma followed by space
        #TODO: Should we somehow force it to generate all alternatives?
        #TODO: No, just to mention that we hope that people are consequent?
        (re.compile(r'([^ \t\n\r\f\v\.\,]*?),\ ([^ \t\n\r\f\v\.\,]*)'),
            r'\2 \1'),

        #TODO: Remove activity from the end

        # This has to be applied last
        (re.compile(r'\ {2,}'), ' '),
        )

#TODO: Shall we have any of our own?
#TODO: How about dashes? http://en.wikipedia.org/wiki/Dash
#TODO: And curly brackets? http://en.wikipedia.org/wiki/Bracket
RULES = ()
###

def generate_variants(text):
    '''
    TODO: Make it a true generator?
    Does not return the given text
    '''
    
    def generate(text, rule, replacement):

        variant = rule.sub(replacement, text)
        if text != variant:
            return variant
        else:
            return None

    variants = [text]
    for rule, replacement in PAPER_RULES:
        for variant in variants:
            new_variant = generate(variant, rule, replacement)
            if new_variant:
                variants.append(new_variant)
    return set(variants[1:])

if __name__ == '__main__':
    from sys import stdin

    for line in (l.rstrip('\n') for l in stdin):
        acc, name = line.split(None, 1)

        for variant in generate_variants(name):
            if variant != name:
                print '{0}\t{1}'.format(acc, variant)
