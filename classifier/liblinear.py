'''
Generic LibLinear classifier class.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-04-04
'''

from pprint import pprint
from math import sqrt
from sys import path as sys_path
from sys import stderr

#from naive import Classifier, CouldNotClassifyError, ClassifierNotTrainedError
#from features import FEATURE_CLASSES
from os.path import dirname
from os.path import join as path_join

sys_path.append(path_join(dirname(__file__), '../'))

from naive import Classifier

from toolsconf import LIBLINEAR_PYTHON_DIR
sys_path.append(LIBLINEAR_PYTHON_DIR)

from linearutil import predict as liblinear_predict
from linearutil import load_model as liblinear_load_model
from linearutil import save_model as liblinear_save_model
from linearutil import train as liblinear_train
from linearutil import problem as liblinear_problem 
from linearutil import parameter as liblinear_parameter

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
    return liblinear_train(liblinear_problem(lbls, vecs),
            liblinear_parameter('-q -s {} -c {}'.format(model_type, c)))
    
def _liblinear_classify(vecs, model):
    import sys
    orig_stdout = sys.stdout
    try:
        with open('/dev/null', 'w') as dev_null:
            sys.stdout = dev_null
            # We don't really need -b 1
            predictions, _, _ = liblinear_predict(
                    [], vecs, model, '')#'-b 1')
            return predictions
    finally:
        sys.stdout = orig_stdout

from random import sample

def _k_folds(k, coll):
    seen = set()
    to_see = set(coll)
    coll_size = len(to_see)
    for i in xrange(k):
        fold_size = int(coll_size / k)
        if i == (k - 1) and coll_size % 2 != 0:
            fold_size -= 1
            
        fold = set(sample(to_see, fold_size))
        seen = seen.union(fold)
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
        return self.vec_index_by_feature_id[feature_id]

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
        self.model = liblinear_train(liblinear_problem(lbls, vecs),
                liblinear_parameter(
                    '-q -s {} -c {}'.format(model_type, 2 ** c)))

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
      
    def _classify(self, vec):
        return self._liblinear_classify(vec)

    def _liblinear_classify(self, vec):
        import sys
        orig_stdout = sys.stdout
        try:
            with open('/dev/null', 'w') as dev_null:
                sys.stdout = dev_null
                # We don't really need -b 1
                predictions, _, _ = liblinear_predict(
                        [], [vec], self.model, '')#'-b 1')
        finally:
            sys.stdout = orig_stdout

        try:
            return self.name_by_lbl_id[predictions[0]]
        except KeyError:
            print predictions
            print self.name_by_lbl_id
            raise

    def classify(self, document, sentence, annotation):
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

        return self._liblinear_classify(vec)

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