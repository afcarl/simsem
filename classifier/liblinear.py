'''
Generic LibLinear classifier class.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-04-04
'''

from math import sqrt
from os import remove
from pprint import pprint
from random import sample
from sys import path as sys_path
from sys import stderr
from tempfile import NamedTemporaryFile

#from naive import Classifier, CouldNotClassifyError, ClassifierNotTrainedError
#from features import FEATURE_CLASSES
from os.path import dirname
from os.path import join as path_join

sys_path.append(path_join(dirname(__file__), '../'))

from naive import Classifier

from toolsconf import LIBLINEAR_PYTHON_DIR
sys_path.append(LIBLINEAR_PYTHON_DIR)

#XXX: If these are imported and we then go into multiprocessing
#   we are in a world of hurt! ctypes and multiprocessing don't mix well
#from linearutil import predict as liblinear_predict
#from linearutil import load_model as liblinear_load_model
#from linearutil import save_model as liblinear_save_model
#from linearutil import train as liblinear_train
#from linearutil import problem as liblinear_problem 
#from linearutil import parameter as liblinear_parameter

MODEL_SEARCH = False
C_SEARCH = True
C_VERBOSE = False
NORMALISE = False

# NOTE: Set to none to disable
#TRAIN_DUMP_FILE_PATH = 'training_data.txt'
#CLASSIFY_DUMP_FILE_PATH = 'classification_data.txt'
TRAIN_DUMP_FILE_PATH = None
CLASSIFY_DUMP_FILE_PATH = None

def _liblinear_train(lbls, vecs, model_type, c):
    from linearutil import (train as liblinear_train,
            problem as liblinear_problem, parameter as liblinear_parameter)
    return liblinear_train(liblinear_problem(lbls, vecs),
            liblinear_parameter('-q -s {0} -c {1}'.format(model_type, c)))
    
def _liblinear_classify(vecs, model):
    from linearutil import predict as liblinear_predict
    import sys
    orig_stdout = sys.stdout
    try:
        with open('/dev/null', 'w') as dev_null:
            sys.stdout = dev_null
            # We don't really need -b 1 here
            predictions, _, _ = liblinear_predict(
                    [], vecs, model, '') #'-b 1')
            return predictions
    finally:
        sys.stdout = orig_stdout

def _k_folds(k, coll):
    to_see = set(coll)
    coll_size = len(to_see)
    for i in xrange(k):
        fold_size = int(coll_size / k)
        if i == (k - 1) and coll_size % 2 != 0:
            fold_size -= 1
            
        fold = set(sample(to_see, fold_size))
        to_see = to_see - fold
        yield fold

from itertools import izip

class hashabledict(dict):
    def __hash__(self):
        try:
            return self.cached_hash
        except AttributeError:
            self.cached_hash = hash(tuple(sorted(self.items())))
        return self.cached_hash

def _dict_hashable(self):
    return hash(tuple(sorted(self.items())))

from itertools import chain

def _c_search(lbls, vecs, model_type=0, k=5):
    vecs = [hashabledict(v) for v in vecs]

    folds = [f for f in _k_folds(k, zip(lbls, vecs))]
    best_c = None
    best_score = -4711
    for c in range(-2, 15, 2):
        if C_VERBOSE:
            print >> stderr, 'Curr C:', c
        scores = []
        for hold_out_fold in folds:
            train_data = set()
            for fold in folds:
                if fold != hold_out_fold:
                    train_data = train_data.union(fold)

            it_lbls = [lbl for lbl, _ in  chain(train_data)]
            it_vecs = [vec for _, vec in chain(train_data)]
            it_model = _liblinear_train(it_lbls, it_vecs, model_type, 2 ** c)

            tp = 0
            fp = 0
           
            predictions = _liblinear_classify(it_vecs, it_model)
            for lbl, prediction in izip(it_lbls, predictions):
                if lbl == prediction:
                    tp += 1
                else:
                    fp += 1

            if fp == 0:
                it_score = 1.0
            else:
                it_score = float(tp) / fp

            scores.append(it_score)

        score = sum(scores) / len(scores)
        if score > best_score:
            best_c = c
            best_score = score
    return best_c

def _texturise(feature_str):
    return feature_str.replace(' ', ':SPACE:')

class LibLinearClassifier(Classifier):
    def __init__(self):
        self.vec_index_by_feature_id = {}
        self.feature_id_by_vec_index = {}
        self.lbl_id_by_name = {}
        self.name_by_lbl_id = {}

        if CLASSIFY_DUMP_FILE_PATH is not None:
            print >> stderr, 'Will write classification data to:', CLASSIFY_DUMP_FILE_PATH
            self.classify_dump_file = open(CLASSIFY_DUMP_FILE_PATH, 'w')
        else:
            self.classify_dump_file = None

    def get_feature(self):
        try:
            return self.feature
        except AttributeError:
            pass

        try:
            self.feature = self.feature_class()
        except AttributeError:
            assert False, ('SubClass of LibLinearClassifier not implementing '
                    'feature_class attribute')

        return self.feature
   
    def _get_lbl_id(self, lbl_name):
        try:
            return self.lbl_id_by_name[lbl_name]
        except KeyError:
            new_lbl_id = len(self.lbl_id_by_name) + 1
            self.lbl_id_by_name[lbl_name] = new_lbl_id
            self.name_by_lbl_id[new_lbl_id] = lbl_name
        return self.lbl_id_by_name[lbl_name]
        
    def _get_vec_index(self, feature_id):
        try:
            return self.vec_index_by_feature_id[feature_id]
        except KeyError:
            new_vec_index = len(self.vec_index_by_feature_id) + 1
            self.vec_index_by_feature_id[feature_id] = new_vec_index
            self.feature_id_by_vec_index[new_vec_index] = feature_id
        return self.vec_index_by_feature_id[feature_id]

    def lbl_vec_to_str(self, lbl, vec):
        return '%s: %s' % (self.name_by_lbl_id[lbl], ' '.join(
            '%s:%s' % (self.feature_id_by_vec_index[k], v)
            for k, v in vec.iteritems()))

    def _train(self, lbls, vecs):
        return self._liblinear_train(lbls, vecs)

    def _liblinear_train(self, lbls, vecs):
        if MODEL_SEARCH:
            # XXX: Will give BOTH c and model...
            raise NotImplementedError
        else:
            model_type = 0
            
        # Find C
        if C_SEARCH:
            c = _c_search(lbls, vecs, model_type=model_type)
            if C_VERBOSE:
                print >> stderr, 'Found C:', c
        else:
            c = 1

        # Train the model
        from linearutil import (train as liblinear_train,
                problem as liblinear_problem,
                parameter as liblinear_parameter)
        self.model = liblinear_train(liblinear_problem(lbls, vecs),
                liblinear_parameter(
                    '-q -s {0} -c {1}'.format(model_type, 2 ** c)))

    def _gen_lbls_vecs(self, documents):
        lbls = []
        vecs = []
        
        if TRAIN_DUMP_FILE_PATH is not None:
            dump_file = open(TRAIN_DUMP_FILE_PATH, 'w')
        else:
            dump_file = None

        for document in documents:
            for sentence in document:
                for annotation in sentence:
                    if dump_file is not None:
                        dump_file.write('!!! ' + sentence.text + '\n')
                        dump_file.write(annotation.type + ':')

                    lbls.append(self._get_lbl_id(annotation.type))
                    vec = {}
                    for f_id, f_val in self.get_feature().featurise(
                            document, sentence, annotation):
                        if dump_file is not None:
                            dump_file.write(' ' + _texturise(f_id)
                                    + ':' + str(f_val))
                        vec[self._get_vec_index(f_id)] = f_val

                    if dump_file is not None:
                        dump_file.write('\n')

                    if NORMALISE:
                        vec_length = float(sqrt(
                                sum(v ** 2 for v in vec.itervalues())))
                        for k in vec:
                            vec[k] = vec[k] / vec_length
                    vecs.append(vec)

        if dump_file is not None:
            print >> stderr, 'Training data dumped to:', TRAIN_DUMP_FILE_PATH
    
        #pprint(self.vec_index_by_feature_id)
        #assert False
        return lbls, vecs

    def train(self, documents):

        lbls, vecs = self._gen_lbls_vecs(documents)

        #print 'Trained on:', len(lbls), len(vecs)
        #print lbls, vecs
        '''
        with open('baren.vecs', 'wa') as baren:
            for lbl, vec in zip(lbls, vecs):
                baren.write(str(lbl) + ' : ' + str(vec))
        '''

        # Train the model
        self._liblinear_train(lbls, vecs)
      
    def _classify(self, vec, ranked=False):
        return self._liblinear_classify(vec, ranked=ranked)

    def _liblinear_classify(self, vec, ranked=False):
        import sys
        orig_stdout = sys.stdout
        try:
            with open('/dev/null', 'w') as dev_null:
                sys.stdout = dev_null

                # Ask for probs. when it is necessary
                if not ranked:
                    args = ''
                else:
                    args = '-b 1'

                from linearutil import predict as liblinear_predict
                predictions, _, probabilities = liblinear_predict(
                        [], [vec], self.model, args)
        finally:
            sys.stdout = orig_stdout

        try:
            if not ranked:
                return self.name_by_lbl_id[predictions[0]]
            else:
                # We need a bit of magic to get this list right since LibLinear
                # only returns a single label we will played with the indexes
                # NOTE: Labels are stored in self.model.label, never assume otherwise
                probs_and_lbl = [(prob, self.name_by_lbl_id[lbl_id])
                        for lbl_id, prob in izip(self.model.label, probabilities[0])]
                probs_and_lbl.sort()
                probs_and_lbl.reverse()
                # Now flip the tuples and we have the ranked labels
                return [(lbl, prob) for prob, lbl in probs_and_lbl]
        except KeyError:
            print predictions
            print self.name_by_lbl_id
            raise

    def classify(self, document, sentence, annotation, ranked=False):
        if (self.vec_index_by_feature_id is None
                or self.lbl_id_by_name is None
                or self.name_by_lbl_id is None):
            raise ClassifierNotTrainedError
        
        if self.classify_dump_file is not None:
            self.classify_dump_file.write(annotation.type + ':')

        vec = {}
        for f_id, f_val in self.get_feature().featurise(
                document, sentence, annotation):

            if self.classify_dump_file is not None:
                self.classify_dump_file.write(' '
                        + _texturise(f_id) + ':' + str(f_val))

            vec[self._get_vec_index(f_id)] = f_val

        if self.classify_dump_file is not None:
            self.classify_dump_file.write('\n')

            vec[self._get_vec_index(f_id)] = f_val

        return self._liblinear_classify(vec, ranked=ranked)

    def __getstate__(self):
        from linearutil import save_model as liblinear_save_model
        # Turn ourselves into a dictionary to pickle
        odict = self.__dict__.copy() # copy the dict since we change it

        # LibLinear requires some trickery
        if 'model' in odict:
            tmp_file = None
            try:
                tmp_file = NamedTemporaryFile('w', delete=False)
                tmp_file.close()

                liblinear_save_model(tmp_file.name, self.model)

                # Replace the old model with the file contents
                with open(tmp_file.name, 'r') as model_data:
                    odict['model'] = model_data.read()
            finally:
                if tmp_file is not None:
                    tmp_file.close()
                    remove(tmp_file.name)

        return odict

    def __setstate__(self, odict):
        from linearutil import load_model as liblinear_load_model
        # Restore ourselves from a dictionary
        self.__dict__.update(odict)

        # LibLinear requires some more trickery
        if 'model' in odict:
            tmp_file = None
            try:
                tmp_file = NamedTemporaryFile('w', delete=False)
                tmp_file.write(self.model)
                tmp_file.close()

                # Replace the string model with the real model
                self.model = liblinear_load_model(tmp_file.name)
            finally:
                if tmp_file is not None:
                    tmp_file.close()
                    remove(tmp_file.name)

if __name__ == '__main__':
    raise NotImplementedError

    # TODO: Get a simple feature, set and test

    '''
    from reader.bionlp import get_id_set
   
    for one_feature_classifier_class in ONE_FEATURE_CLASSIFIERS:
        train, dev, _ = get_id_set()
        classifier = one_feature_classifier_class()
        classifier.train(train)

        for document in dev:
            for sentence in document:
                for annotation in sentence:
                    classifier.classify(document, sentence, annotation)
    '''
