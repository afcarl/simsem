#!/usr/bin/env python

'''
Train a SimSem model.

Get some data (any ST-format really):

    find data/corpora/bionlp_2011_st/*genia* -name '*.a1' -o -name '*.a2' \
            | xargs -r cat | grep '^T' | cut -f 2,3 \
            | sed -e 's|\(.*\)\ [0-9]\+\ [0-9]\+\t\(.*\)|\2\t\1|g'

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-22
'''

# XXX: Quick hack, needs model/dataset selection etc.
# XXX: Assumes no context and <STRING>\t<TYPE> format
from argparse import ArgumentParser, FileType

try:
    from cPickle import dump as pickle_dump
except ImportError:
    from cPickle import dump as pickle_dump

from resources import Annotation, Sentence, Document
from classifier.competitive import SimStringInternalClassifier

# TODO: Should not be done this way, API for caching
from experiment.common import cache_simstring

### Constants
ARGPARSER = ArgumentParser(description='XXX')#XXX: TODO:
ARGPARSER.add_argument('model_path')
ARGPARSER.add_argument('-v', '--verbose', action='store_true')
ARGPARSER.add_argument('-i', '--input', default='-', type=FileType('r'),
        help='input source (DEFAULT: stdin)')
###

def _tab_separated_input_to_doc(input):
    # Create a dataset out of the input
    doc = Document(input.name, [], [], '<%s>' % input.name)
    for _string, _type in (l.rstrip('\n').split('\t') for l in input):
        doc.abstract.append(Sentence(_string,
            [Annotation(0, len(_string), _type), ]))
    return doc

def main(args):
    argp = ARGPARSER.parse_args(args[1:])

    # Create a dataset out of the input
    doc = _tab_separated_input_to_doc(argp.input)

    # Cache the strings for speed
    cache_simstring(((doc, ), ), verbose=argp.verbose)

    classifier = SimStringInternalClassifier()
    classifier.train((doc, ))

    with open(argp.model_path, 'w') as model_file:
        pickle_dump(classifier, model_file)

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
