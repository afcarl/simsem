#!/usr/bin/env python

'''
Classify strings using a SimSem model.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-22
'''

# XXX: Assumes no context and <STRING>\t<TYPE> format
from argparse import ArgumentParser, FileType

try:
    from cPickle import load as pickle_load
except ImportError:
    from pickle import load as pickle_load

from classifier.competitive import SimStringInternalClassifier
from resources import Document, Sentence, Annotation

# TODO: Should not be done this way, API for caching
from experiment.common import cache_simstring
from train import _tab_separated_input_to_doc

### Constants
ARGPARSER = ArgumentParser(description='XXX')#XXX: TODO:
ARGPARSER.add_argument('model_path')
ARGPARSER.add_argument('-v', '--verbose', action='store_true')
ARGPARSER.add_argument('-n', '--no-cache', action='store_true')
ARGPARSER.add_argument('-i', '--input', default='-', type=FileType('r'),
        help='input source (DEFAULT: stdin)')
###

def _string_to_ann_sent(_string):
    return Sentence(_string, [Annotation(0, len(_string), None)])

def main(args):
    argp = ARGPARSER.parse_args(args[1:])

    if not argp.no_cache:
        # We can't do it iteratively listening to stdin, read it all
        doc = Document('<classify>', [], [], '<classify>')
        for _string in (l.rstrip('\n') for l in argp.input):
            doc.abstract.append(_string_to_ann_sent(_string))
        docs = (doc, )
    else:
        docs = (Document('Line: %s' % i, [], [_string_to_ann_sent(_string)],
            '<stdin>') for  i, _string in enumerate(
                (l.rstrip('\n') for l in argp.input), start=1))

    # Cache the strings for speed
    if not argp.no_cache:
        cache_simstring((docs, ), verbose=argp.verbose)

    with open(argp.model_path, 'r') as model_file:
        classifier = pickle_load(model_file)

    # TODO: Faster to do it in a batch instead
    for doc in docs:
        for sent in doc:
            for ann in sent:
                print '%s\t%s' % (sent.annotation_text(ann),
                        str(classifier.classify(doc, sent, ann, ranked=True)))

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
