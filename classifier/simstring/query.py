'''
Querying a SimString db and subsequent caching.

Author:     Pontus Stenetorp    <pontus is s u-tokyo ac jp>
Version:    2011-04-09
'''

# TODO: CHECKSUMS
# TODO: Checksums for the pickle too, we also need a file lock
# TODO: ONLY load the relevant dbs, multiple pickles

from string import digits
from itertools import imap, groupby, combinations
from os.path import join as path_join
from os.path import dirname, basename, isfile, normpath
from os import listdir
from re import compile as re_compile
from atexit import register as atexit_register
from sys import path as sys_path

try:
    from cPickle import load as pickle_load
    from cPickle import dump as pickle_dump
except ImportError:
    from pickle import load as pickle_load
    from pickle import dump as pickle_dump

sys_path.append(path_join(dirname(__file__), '..', '..'))

#from features import Feature

from toolsconf import SIMSTRING_LIB_PATH
sys_path.insert(0, SIMSTRING_LIB_PATH)

from simstring import reader as simstring_reader
from simstring import cosine as simstring_cosine

### XXX:
#from classifier.liblinear import LibLinearClassifier

### Constants
from config import SIMSTRING_DB_PATHS

CDB_REGEX = re_compile(r'\.[0-9]+\.cdb$')
# XXX: Should check for a variable in config for the directory
SIMSTRING_QUERY_CACHE_DIR_PATH = normpath(path_join(dirname(__file__),
        '../../cache'))
SIMSTRING_QUERY_CACHE_PATH = path_join(SIMSTRING_QUERY_CACHE_DIR_PATH,
        'simstring_query.cache')

# Cut-off for the amount of data to process for a single query
# NOTE: Must be False or a positive integer
RESPONSE_CUT_OFF = 10
# NOTE: This is a hard cut, nothing will ever go below it
QUERY_CUT_OFF = 3
###

### SimString Cache
# Defer loading
global SIMSTRING_QUERY_CACHE
SIMSTRING_QUERY_CACHE = None
global MODIFIED_SIMSTRING_QUERY_CACHE
MODIFIED_SIMSTRING_QUERY_CACHE = False

def _simstring_db_checksum(db_path):
    raise NotImplementedError
    # XXX: Should do a checksum for the db! and CDB
    # turku_event_corpus_triggers.transcription.db.32.cdb
    #raise NotImplementedError

def _load_simstring_cache():
    global SIMSTRING_QUERY_CACHE
    if not isfile(SIMSTRING_QUERY_CACHE_PATH):
        #XXX: TODO: We need a check-sum test if we are to keep a specific db
        SIMSTRING_QUERY_CACHE = {}
    else:
        with open(SIMSTRING_QUERY_CACHE_PATH, 'rb') as cache_file:
            SIMSTRING_QUERY_CACHE = pickle_load(cache_file)

from os.path import exists

# Upon exiting, save the cache
@atexit_register
def _save_simstring_query_cache():
    # Check if the cache directory exists, otherwise create it
    if not exists(dirname(SIMSTRING_QUERY_CACHE_DIR_PATH)):
        from os import makedirs
        makedirs(SIMSTRING_QUERY_CACHE_DIR_PATH)

    # Save if we have a cache and it has been modified
    if SIMSTRING_QUERY_CACHE is not None and MODIFIED_SIMSTRING_QUERY_CACHE:
        with open(SIMSTRING_QUERY_CACHE_PATH, 'wb') as cache_file:
            # Dump with highest protocol
            pickle_dump(SIMSTRING_QUERY_CACHE, cache_file, -1)

#XXX: Fixed measure, can't alter it
#XXX: Stupid name reader arg, use_reader... cached_reader
def query_simstring_db(query, db_path, reader_arg=None):
    global SIMSTRING_QUERY_CACHE
    global MODIFIED_SIMSTRING_QUERY_CACHE
    if SIMSTRING_QUERY_CACHE is None:
        _load_simstring_cache()

    try:
        cache = SIMSTRING_QUERY_CACHE[db_path]
    except KeyError:
        cache = {}
        SIMSTRING_QUERY_CACHE[db_path] = cache
        MODIFIED_SIMSTRING_QUERY_CACHE = True

    try:
        return cache[query]
    except KeyError:
        MODIFIED_SIMSTRING_QUERY_CACHE = True

    # We have to query this...
    #assert False, 'NOT ALLOWED TO QUERY!'

    if reader_arg is None:
        reader = None
    try:
        if reader_arg is None:
            reader = simstring_reader(db_path)
        else:
            reader = reader_arg
        
        reader.measure = simstring_cosine
        for threshold in (v / 10.0 for v in xrange(10, QUERY_CUT_OFF - 1, -1)):
            reader.threshold = threshold

            # The reader will choke on unicode objects, so encode it
            query_utf8 = query.encode('utf-8')
            response = reader.retrieve(query_utf8)

            if not TSURUOKA_DIST:
                # Only save whether we got a response or not
                if response:
                    response = True
                else:
                    response = False
                tsuruoka_dist = None
            else:
                # Okay, now we are in a pickle, SimString has returned
                # everything sorted by length... Although it had it internally
                # by n-gram. *sigh* We need it by n-gram.

                if response:
                    # Sort the response to prepare a cut-off
                    from lib.ngram import n_gram_ref_cos_cmp, n_gram_gen
                    ref_grams = set(g for g in n_gram_gen(query, n=3,
                        guards=TSURUOKA_GUARDED))

                    # We need Unicode internally at this point
                    response = [s.decode('utf-8') for s in response]
                    response = sorted(response,
                            cmp=lambda a, b: -n_gram_ref_cos_cmp(
                                a, b, ref_grams, guards=TSURUOKA_GUARDED))
                    # Cut-off time!
                    response = response[:RESPONSE_CUT_OFF]

                    if TSURUOKA_NORMALISED:
                        tsuruoka_dist = max(bucket_norm_tsuruoka(query, resp_str)
                                for resp_str in response)
                    else:
                        tsuruoka_dist = min(bucket_tsuruoka(query, resp_str)
                                for resp_str in response)
            if response:
                cache[query] = (threshold, tsuruoka_dist)
                # We can and should bail at this point
                break
        else:
            # We found no results for any threshold
            cache[query] = (None, None)
    finally:
        # Only close if we were not passed the reader
        if reader_arg is None and reader is not None:
            reader.close()
    
    #print cache
    #print SIMSTRING_CACHE_BY_DB[db_path]
    #print SIMSTRING_CACHE_BY_DB
    #if len(cache) > 100:
    #    exit(-1)
    return cache[query]

# TODO: Move the code below somewhere suitable

from itertools import chain
from sys import maxint

from lib.sdistance import tsuruoka, tsuruoka_norm

TSURUOKA_BUCKETS = tuple(chain(xrange(0, 100, 10), xrange(100, 1000, 50),
    xrange(1000, 10001, 1000), (maxint, )))
# NOTE: Not optimal but should do the trick for now
def _bucket(num):
    for bucket in TSURUOKA_BUCKETS:
        if num <= bucket:
            return bucket
    else:
        assert False, 'no bucket found, larger than maxint? really?'

TSURUOKA_NORM_BUCKETS = tuple((x / 100.0 for x in xrange(100, -1, -5)))
# NOTE: Not optimal but should do the trick for now
def _norm_bucket(num):
    for bucket in TSURUOKA_NORM_BUCKETS:
        if num <= bucket:
            return bucket
    else:
        assert False, 'no bucket found, was that really normalised?'

def bucket_tsuruoka(a, b):
    return _bucket(tsuruoka(a, b))

def bucket_norm_tsuruoka(a, b):
    return _norm_bucket(tsuruoka_norm(a, b))

### Constants ###
TSURUOKA_GUARDED = True
TSURUOKA_DIST = True
TSURUOKA_NORMALISED = True
###
