'''
Abstract SimString package classes.

Author:     Pontus Stenetorp    <pontus is s u-tokyo ac jp>
Version:    2011-04-09
'''

from os.path import join as path_join
from os.path import basename, dirname
from sys import path as sys_path

from config import SIMSTRING_DB_PATHS, SIMSTRING_DB_DIR
from query import query_simstring_db

sys_path.append(path_join(dirname(__file__), '..', '..'))

from ..liblinear import LibLinearClassifier

from toolsconf import SIMSTRING_LIB_PATH
sys_path.append(SIMSTRING_LIB_PATH)

from simstring import reader as simstring_reader
from simstring import cosine as simstring_cosine


class AbstractSimStringClassifier(LibLinearClassifier):
    def __init__(self):
        raise NotImplementedError

    def get_feature(self):
        try:
            return self.feature
        except AttributeError:
            pass

        try:
            self.feature = self.feature_class()
        except AttributeError:
            assert False, ('SubClass of SimStringClassifier not '
                    'implementing feature_class attribute')

        return self.feature

    def classify(self, document, sentence, annotation):
        feature = self.get_feature()

    def train(self, documents):
        feature = self.get_feature()
        raise NotImplementedError


class AbstractSimStringFeature(object): #XXX: FIX INHERIT (Feature):
    def __init__(self):
        raise NotImplementedError

    def _db(self):
        try:
            return self._db_path
        except AttributeError:
            # TODO: This really only needs to run ONCE! Set a field!
            for db_path in SIMSTRING_DB_PATHS:
                # TODO: Should be a split
                try:
                    if db_path.endswith(self.db_name):
                        self._db_path = db_path
                        break
                except AttributeError:
                    assert False, ('SubClass of SimStringFeature not '
                            'implementing db_name attribute')
            else:
                print SIMSTRING_DB_PATHS
                assert False, ('{0} not found in {1}'
                        ).format(self.db_name, SIMSTRING_DB_DIR)
        return self._db_path

    def get_id(self):
        # XXX: Hack for speed-up
        try:
            return self._id
        except AttributeError:
            self._id = 'SimStringFeature<{0}>'.format(basename(self._db()))
        return self._id

    def featurise(self, document, sentence, annotation):
        query_text = sentence.annotation_text(annotation)
        result = query_simstring_db(query_text, self._db())

        if result is not None:
            yield (str(result), 1)
