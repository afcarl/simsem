#!/usr/bin/env python

'''
Train a SimSem model.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-22
'''

# XXX: Quick hack, needs model/dataset selection etc.
# XXX: Assumes no context and <STRING>\t<TYPE> format
from argparse import ArgumentParser, FileType

from resources import Annotation, Sentence, Document
from classifier.competitive import SimStringInternalClassifier

# TODO: Should not be done this way, API for caching
from test import _cache_simstring

### Constants
ARGPARSER = ArgumentParser(description='XXX')#XXX: TODO:
ARGPARSER.add_argument('model_path')
ARGPARSER.add_argument('-v', '--verbose', action='store_true')
ARGPARSER.add_argument('-i', '--input', default='-', type=FileType('r'),
        help='input source (DEFAULT: stdin)')
###

def main(args):
    argp = ARGPARSER.parse_args(args[1:])

    # Create a dataset out of the input
    doc = Document('stdin', [], [], '<stdin>')
    for _string, _type in (l.rstrip('\n').split('\t') for l in argp.input):
        doc.abstract.append(Sentence(_string,
            [Annotation(0, len(_string), _type), ]))
    docs = (doc, ) # The API generally deals with collections of documents

    # Cache the strings for speed
    _cache_simstring((docs, ), verbose=argp.verbose)

    classifier = SimStringInternalClassifier()
    classifier.train(docs)

    from linearutil import save_model as liblinear_save_model
    liblinear_save_model(argp.model_path, classifier.model)

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
