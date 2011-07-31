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
# XXX: TODO: More hyphen variants here?
TSURUOKA_2004_INS_DEL_CHEAP = set((' ', '-'))
DIGITS = set(digits)

from config import SIMSTRING_DB_PATHS

CDB_REGEX = re_compile(r'\.[0-9]+\.cdb$')
# XXX: Should check for a variable in config for the directory
SIMSTRING_QUERY_CACHE_DIR_PATH = normpath(path_join(dirname(__file__),
        '../../cache'))
SIMSTRING_QUERY_CACHE_PATH = path_join(SIMSTRING_QUERY_CACHE_DIR_PATH,
        'simstring_query.cache')
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
        for threshold in (v / 10.0 for v in xrange(10, 0, -1)):
            reader.threshold = threshold
            try:
                # The reader will choke on unicode objects, so encode it
                if reader.retrieve(query.encode('utf-8')):
                    cache[query] = threshold
    
                    if reader_arg is None:
                        reader.close()

                    # We can bail at this point
                    break
                else:
                    cache[query] = None
            except TypeError:
                print type(query)
                raise
    finally:
        # Only close if we were given the reader
        if reader_arg is None and reader is not None:
            reader.close()
    
    #print cache
    #print SIMSTRING_CACHE_BY_DB[db_path]
    #print SIMSTRING_CACHE_BY_DB
    #if len(cache) > 100:
    #    exit(-1)
    return cache[query]
