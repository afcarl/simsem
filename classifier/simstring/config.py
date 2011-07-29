'''
Configurations for the SimSem classifier.

Author:     Pontus Stenetorp    <pontus is s u-tokyo ac jp>
Version:    2011-04-03
'''

from os.path import join as path_join
from os.path import dirname, basename, isfile
from os import listdir

### Constants
SIMSTRING_DB_DIR = path_join(dirname(__file__),
        '..', '..', 'data', 'simstring', 'dbs')
SIMSTRING_DB_PATHS = [path_join(SIMSTRING_DB_DIR, p)
        for p in listdir(SIMSTRING_DB_DIR) if p.endswith('.db')]
FEATURES_MODULE_PATH = path_join(dirname(__file__), 'features.py')
CLASSIFIERS_MODULE_PATH = path_join(dirname(__file__), 'classifiers.py')
###
