'''
Shared functionality between experimental modules.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-30
'''

from itertools import izip
from sys import stderr

from simstring import reader as simstring_reader

from classifier.simstring.config import SIMSTRING_DB_PATHS
from classifier.simstring.query import query_simstring_db

# I hate Python 2.6...
def compress(it, flter):
    for e, v in izip(it, flter):
        if v:
            yield e

# XXX: Pre-load the simstring cache uglily!
def cache_simstring(datasets, verbose=False, ann_modulo=1000,
        queries_modulo=1000):
    if verbose:
        print >> stderr, 'Caching SimString:'

        print >> stderr, 'Pre-caching queries...',
        queries_seen = 0

    # For most cases we are better off caching every single query instead of
    # iterating over them, this also makes sure that each query is unique when
    # we finally hit the SimString database
    queries = set()
    for dataset in datasets:
        for document in dataset:
            for sentence in document:
                for annotation in sentence:
                    queries.add(sentence.annotation_text(annotation))
                    if verbose:
                        queries_seen += 1
                        if queries_seen % queries_modulo == 0:
                            print >> stderr, queries_seen, '...',
    if verbose:
        print >> stderr, ('Done! (reduced from {} to {})'
                ).format(queries_seen, len(queries))

    for db_i, db_path in enumerate(SIMSTRING_DB_PATHS, start=1):
        if verbose:
            print >> stderr, 'Caching for db: {0} ({1}/{2}) ...'.format(db_path, db_i,
                    len(SIMSTRING_DB_PATHS)),
        
        if verbose:
            ann_cnt = 0
        db_reader = None
        try:
            db_reader = simstring_reader(db_path)
            for query in queries:
                query_simstring_db(query, db_path,
                    reader_arg=db_reader)

                if verbose:
                    ann_cnt += 1
                    if ann_cnt % ann_modulo == 0:
                        print >> stderr, ann_cnt, '...',

        finally:
            if db_reader is not None:
                db_reader.close()
        if verbose:
            print >> stderr, 'Done!'

def simstring_caching(classifiers, document_sets, verbose=False):
    if any((True for k in classifiers
        # NOTE: Keep this check up to date
        if 'SIMSTRING' in k or 'GAZETTER' in k or 'TSURUOKA' in k)):
        if verbose:
            print >> stderr, 'Caching queries for SimString:'

        cache_simstring(document_sets, verbose=verbose)
    else:
        if verbose:
            print >> stderr, 'No caching necessary for the given classifier'
