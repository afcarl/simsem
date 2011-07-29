'''
SimSem classifiers module.

Author:     Pontus Stenetorp    <pontus is s u-tokyo ac jp>
Version:    2011-04-07
'''

from os.path import join as path_join
from os.path import basename, dirname, isfile
from sys import path as sys_path

# XXX: Path hack!
sys_path.append(path_join(dirname(__file__), '..'))
from liblinear import LibLinearClassifier
#from ....features import AbstractFeature

### Code generation and import of generated code

# XXX: The below code is NOT extensively tested
# TODO: Can this be inserted into __init__.py? If so we can expose
#       "classifiers" outside of the package
#XXX: If regeneration doesn't work it is most likely a more serious error, print it!

DEBUG = False

from config import (SIMSTRING_DB_PATHS, FEATURES_MODULE_PATH,
        CLASSIFIERS_MODULE_PATH)

def _features_module_is_valid():
    try:
        from features import SIMSTRING_FEATURES
        if DEBUG:
            print 'Import OK'
        return len(SIMSTRING_FEATURES) == len(SIMSTRING_DB_PATHS)
    except ImportError, e:
        #raise
        if DEBUG:
            #print e
            print 'Import X'
        return False

def _generate_features_module():
    from generate import simstring_features_module_template

    with open(FEATURES_MODULE_PATH, 'w') as features_module_file:
        features_module_file.write(simstring_features_module_template(
            basename(p) for p in SIMSTRING_DB_PATHS))

    if not _features_module_is_valid():
        # Probably a more serious error, let it crash
        import features

if not _features_module_is_valid():
    _generate_features_module()

# Classifiers
from features import SIMSTRING_FEATURES

def _classifier_module_is_valid():
    try:
        from classifiers import SIMSTRING_CLASSIFIERS
        if DEBUG:
            print 'Import OK'
        return len(SIMSTRING_FEATURES) == len(SIMSTRING_CLASSIFIERS)
    except ImportError:
        if DEBUG:
            print 'Import X'
        return False

def _generate_classifier_module():
    from generate import simstring_classifiers_module_template

    with open(CLASSIFIERS_MODULE_PATH, 'w') as classifiers_module_file:
        classifiers_module_file.write(simstring_classifiers_module_template(
            SIMSTRING_FEATURES))

    assert _classifier_module_is_valid()

if not _classifier_module_is_valid():
    _generate_classifier_module()

from classifiers import SIMSTRING_CLASSIFIERS
###

SIMSTRING_CUTOFF = 0.7
SIMSTRING_CHAIN = True

# TODO: Filter out dbs
class SimStringEnsembleFeature(object): #XXX: Inherit AbstractFeature!
    def __init__(self):
        self.features = [c() for c in SIMSTRING_FEATURES]

    def get_id(self):
        try:
            return self._id
        except AttributeError:
            # The hash should uniquely identify a set of SimString features
            self._id = 'SimStringEnsembleFeature<0x{:x}>'.format(
                    hash(' '.join(sorted([f.get_id() for f in self.features]))))
        return self._id

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            if SIMSTRING_CUTOFF is None:
                for f_tup in feature.featurise(document, sentence, annotation):
                    yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])
            else:
                for f_tup in feature.featurise(document, sentence, annotation):
                    simstring_confidence = float(f_tup[0])          
                    if simstring_confidence >= SIMSTRING_CUTOFF:
                        yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])
                        # Also generate for all below
                        if SIMSTRING_CHAIN:
                            for confi in range(int(simstring_confidence * 10),
                                    int(SIMSTRING_CUTOFF * 10) - 1, -1):
                                yield (str(confi / 10.0)
                                        + '-(<' + feature.get_id() + '>)', f_tup[1])


'''
class SimStringWindowEnsembleFeature(object): #XXX: Inherit AbstractFeature!
    def __init__(self):
        self.features = [c() for c in SIMSTRING_FEATURES]

    def get_id(self):
        # The hash should uniquely identify a set of SimString features
        return 'SimStringEnsembleFeature<0x{:x}>'.format(
                hash(' '.join(sorted([f.get_id() for f in self.features]))))

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])
'''

class SimStringGazetterEnsembleFeature(object):
    def __init__(self):
        self.features = [c() for c in SIMSTRING_FEATURES]

    def get_id(self):
        try:
            return self._id
        except AttributeError:      
        # The hash should uniquely identify a set of SimString features
            self._id = 'SimStringEnsembleFeature<0x{:x}>'.format(
                   hash(' '.join(sorted([f.get_id() for f in self.features]))))
        return self._id

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                # Only yield direct hits
                if f_tup[0] == '1.0':
                    #print f_tup[0], type(f_tup[0])
                    #assert False
                    yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])


class SimStringEnsembleClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = SimStringEnsembleFeature


class SimStringGazetterEnsembleClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = SimStringGazetterEnsembleFeature 
