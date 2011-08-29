#!/usr/bin/env python

'''
Experimental test-suite for SimSem.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-03-02
'''

from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from copy import deepcopy
from itertools import chain, izip, tee
from json import dumps as json_dumps
from math import sqrt
from multiprocessing import Pool
from operator import itemgetter
from os import listdir, makedirs
from os.path import abspath, basename, dirname, join as join_path
from random import sample, seed, random
from shutil import copy
from sys import path as sys_path, stderr

try:
    from cPickle import dump as pickle_dump, load as pickle_load
except ImportError:
    from pickle import dump as pickle_dump, load as pickle_load

try:
    from collections import OrderedDict
except ImportError:
    # Backported version: http://pypi.python.org/pypi/ordereddict/
    from ordereddict import OrderedDict

from reader.bionlp import (get_epi_set, get_genia_set, get_id_set,
        get_bionlp_2009_set, get_grec_set, get_super_grec_set,
        get_calbc_cii_set, get_nlpba_set, get_nlpba_down_set)
from classifier.naive import NaiveClassifier, MaximumClassifier 
from misc import writeable_dir

from classifier.simstring.classifier import (SimStringEnsembleClassifier,
        SimStringGazetterEnsembleClassifier)
from classifier.simstring.query import query_simstring_db
from classifier.simstring.config import SIMSTRING_DB_PATHS

from toolsconf import SIMSTRING_LIB_PATH
sys_path.append(SIMSTRING_LIB_PATH)

from simstring import reader as simstring_reader

from classifier.competitive import SimpleInternalEnsembleClassifier
from classifier.competitive import CompetitiveEnsembleClassifier
from classifier.competitive import SimStringCompetitiveEnsembleClassifier
from classifier.competitive import SimStringInternalClassifier
from classifier.competitive import GazetterInternalClassifier
from classifier.competitive import InternalClassifier
from classifier.competitive import TsuruokaInternalClassifier
from classifier.competitive import TsuruokaClassifier
from classifier.competitive import SimStringTsuruokaInternalClassifier
from classifier.competitive import SimStringTsuruokaClassifier

from classifier.liblinear import _k_folds, hashabledict


### Constants
CLASSIFIERS = OrderedDict((
    # Rather redundant at this stage
    #('MAXVOTE',                     MaximumClassifier),
    ('NAIVE',                       NaiveClassifier),
    #('SIMPLE-INTERNAL-ENSEMBLE',    SimpleInternalEnsembleClassifier),
    ('INTERNAL',                    InternalClassifier),
    ('SIMSTRING',                   SimStringEnsembleClassifier),
    ('GAZETTER',                    SimStringGazetterEnsembleClassifier),
    #('COMPETITIVE',                 CompetitiveEnsembleClassifier),
    ('INTERNAL-SIMSTRING',          SimStringInternalClassifier),
    ('INTERNAL-GAZETTER',           GazetterInternalClassifier),
    # To use the Tsuruoka classifiers you need query.py to have them enabled
    ('INTERNAL-TSURUOKA',           TsuruokaInternalClassifier),
    ('TSURUOKA',                    TsuruokaClassifier),
    ('INTERNAL-SIMSTRING-TSURUOKA', SimStringTsuruokaInternalClassifier),
    ('SIMSTRING-TSURUOKA',          SimStringTsuruokaClassifier),
    ))

DATASETS = OrderedDict((
    #('BioNLP-ST-2009',                      get_bionlp_2009_set),
    ('BioNLP-ST-2011-Epi_and_PTM',          get_epi_set),
    ('BioNLP-ST-2011-Infectious_Diseases',  get_id_set),
    ('BioNLP-ST-2011-genia',                get_genia_set),
    ('CALBC_CII',                            get_calbc_cii_set),
    #('GREC',                                get_grec_set),
    ('NLPBA',                               get_nlpba_set),
    ('NLPBA-DOWN',                          get_nlpba_down_set),
    ('SUPER_GREC',                          get_super_grec_set),
    ))
###

# I hate Python 2.6...
def _compress(it, flter):
    for e, v in izip(it, flter):
        if v:
            yield e

def _mean(l):
    return sum(l) / float(len(l))

def _median(l):
    return sorted(l)[len(l)/2]

def _truncated_mean(l, truncation=0.05):
    offset = int(len(l) * truncation)
    return sum(sorted(l)[offset:-offset]) / float(len(l) - 2 * offset)

def _score_classifier(classifier, test_set):
    # (TP, FP, FN) # Leaving out TN
    results_by_class = {} #XXX: THIS HAS TO BE A CLASS!

    for document in test_set:
        for sentence in document:
            for annotation in sentence:
                # TODO: Cast annotation into span! It needs to be censored
                predicted = classifier.classify(document, sentence, annotation)
                
                try:
                    results_by_class[annotation.type]
                except KeyError:
                    results_by_class[annotation.type] = (0, 0, 0)

                try:
                    results_by_class[predicted]
                except KeyError:
                    results_by_class[predicted] = (0, 0, 0)

                a_tp, a_fp, a_fn = results_by_class[annotation.type]
                p_tp, p_fp, p_fn = results_by_class[predicted]

                if predicted == annotation.type:
                    results_by_class[annotation.type] = (a_tp + 1, a_fp, a_fn)
                if predicted != annotation.type:
                    results_by_class[annotation.type] = (a_tp, a_fp, a_fn + 1)
                    results_by_class[predicted] =  (p_tp, p_fp + 1, p_fn)

    # Extend the results to:
    # macro, micro, {RESULTS_BY_CLASS}
    tp_sum = sum([tp for tp, _, _ in results_by_class.itervalues()])
    fn_sum = sum([fn for _, _, fn in results_by_class.itervalues()])
    macro_score = tp_sum / float(tp_sum + fn_sum)
    
    micro_scores = []
    for res_tup in results_by_class.itervalues():
        m_tp, _, m_fn = res_tup
        m_tot = float(m_tp + m_fn)
        if m_tot <= 0:
            micro_scores.append(1.0)
        else:
            micro_scores.append(m_tp / float(m_tp + m_fn))
    micro_score = _avg(micro_scores)

    return (macro_score, micro_score, tp_sum, fn_sum, results_by_class)

def _score_classifier_by_tup(classifier, test_tups):
    # (TP, FP, FN) # Leaving out TN
    results_by_class = {} #XXX: THIS HAS TO BE A CLASS!

    for test_lbl, test_vec in izip(*test_tups):
        if not isinstance(test_lbl, str):
            test_lbl_type = classifier.name_by_lbl_id[test_lbl]
        else:
            test_lbl_type = test_lbl

        # TODO: Cast annotation into span! It needs to be censored
        predicted = classifier._classify(test_vec)
        
        try:
            results_by_class[test_lbl_type]
        except KeyError:
            results_by_class[test_lbl_type] = (0, 0, 0)

        try:
            results_by_class[predicted]
        except KeyError:
            results_by_class[predicted] = (0, 0, 0)

        a_tp, a_fp, a_fn = results_by_class[test_lbl_type]
        p_tp, p_fp, p_fn = results_by_class[predicted]

        if predicted == test_lbl_type:
            results_by_class[test_lbl_type] = (a_tp + 1, a_fp, a_fn)
        if predicted != test_lbl_type:
            results_by_class[test_lbl_type] = (a_tp, a_fp, a_fn + 1)
            results_by_class[predicted] =  (p_tp, p_fp + 1, p_fn)

    # Extend the results to:
    # macro, micro, {RESULTS_BY_CLASS}
    tp_sum = sum([tp for tp, _, _ in results_by_class.itervalues()])
    fn_sum = sum([fn for _, _, fn in results_by_class.itervalues()])
    macro_score = tp_sum / float(tp_sum + fn_sum)
    
    micro_scores = []
    for res_tup in results_by_class.itervalues():
        m_tp, _, m_fn = res_tup
        m_tot = float(m_tp + m_fn)
        if m_tot <= 0:
            micro_scores.append(1.0)
        else:
            micro_scores.append(m_tp / float(m_tp + m_fn))
    micro_score = _avg(micro_scores)

    return (macro_score, micro_score, tp_sum, fn_sum, results_by_class)
        
    '''
    true_positive = 0
    false_positive = 0
    
    for document in test_set:
        for sentence in document:
            for annotation in sentence:
                # TODO: Cast annotation into span! It needs to be censored
                predicted = classifier.classify(document, sentence, annotation)
                if predicted == annotation.type:
                    true_positive += 1
                else:
                    false_positive += 1

    return (true_positive, false_positive,
            float(true_positive) / (true_positive + false_positive))
    '''

### Maths
def _avg(vals):
    return sum(vals) / float(len(vals))

def _stddev(vals):
    avg = _avg(vals)
    return sqrt(sum(((val - avg) ** 2 for val in vals)) / len(vals))

### Tests

# XXX: Pre-load the simstring cache uglily!
def _cache_simstring(datasets, verbose=False, ann_modulo=1000,
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

def __learning_curve_test_data_set(args):
    return _learning_curve_test_data_set(*args)


# TODO: We probably need more folds
def _learning_curve_test_data_set(classifiers, dataset_id, dataset_getter,
        verbose=False, no_simstring_cache=False, use_test_set=False, folds=10,
        min_perc=5, max_perc=101, step_perc=5, it_factor=1):
    if verbose:
        print >> stderr, 'Data set:', dataset_id

    if verbose:
        print >> stderr, 'Caching vectorised data...',
    train, dev, test = dataset_getter()
    if use_test_set:
        train, dev = list(chain(train, dev)), list(test)
    else:
        train, dev = list(train), list(dev)

    if verbose:
        print >> stderr, 'Done!'

    if verbose:
        print >> stderr, 'Calculating train set size...',
    train_size = 0
    for d in train:
        for s in d:
            for a in s:
                train_size += 1
    if verbose:
        print >> stderr, 'Done!'

    # Generate train folds
    '''
    for p in xrange(min_perc, max_perc, step_perc):
        train_sets.append([sample(train, int((p / 100.0) * len(train)))
                for _ in xrange(folds)])
    '''
    if verbose:
        print >> stderr, 'Generating filters...',
    train_filters = []
    indices = [i for i in xrange(train_size)]
    for p in xrange(min_perc, max_perc, step_perc):
        sample_size = int((p / 100.0) * train_size)
        # XXX: Heuristic, * 2?

        if it_factor is not None:
            folds = int(int(train_size / float(sample_size)) * it_factor)
        else:
            folds = 1
    
        if max_perc == 100:
            folds = 1
        elif folds < 2:
            folds = 2

        fold_filters = []
        for _ in xrange(folds):
            selected_indices = set(sample(indices, sample_size))
            fold_filters.append([i in selected_indices for i in indices])
        train_filters.append(fold_filters)

    if verbose:
        print >> stderr, 'Done!'

    if not no_simstring_cache:
        _simstring_caching(classifiers, (train, dev), verbose=verbose)

    # Collect the seen type to iterate over later
    seen_types = set()
    results_by_classifier = {}

    for classifier_id, classifier_class in classifiers.iteritems():
        ###
        if verbose:
            print >> stderr, 'Classifier:', classifier_id,
            
        classifier = classifier_class()
        train_lbls, train_vecs = classifier._gen_lbls_vecs(train)
        test_lbls,  test_vecs = classifier._gen_lbls_vecs(dev)

        assert len(train_lbls) == train_size, '{} != {}'.format(
                len(train_lbls), train_size)

        classifier_results = {}
        
        for train_fold_filters in train_filters:
            '''
            if verbose:
                print >> stderr, 'Sample size: ({0}/{1}) ...'.format(
                        -1, train_size),
            '''
            
            scores = []
            for i, train_fold_filter in enumerate(train_fold_filters, start=1):
                if verbose and i % 10 == 0:
                    print >> stderr, i, '...',

                train_fold_lbls = [l for l in _compress(train_lbls, train_fold_filter)]
                train_fold_vecs = [v for v in _compress(train_vecs, train_fold_filter)]
                '''if verbose:
                    print >> stderr, 'Test fold: ({0}/{1})'.format(
                            i, len(train_fold_filter))
                '''

                #classifier = classifier_class()
                #assert not [i for i in train_fold_lbls if i == 0]

                assert train_fold_lbls, train_fold_filter
                assert train_fold_vecs, train_fold_filter
                classifier._train(train_fold_lbls, train_fold_vecs)

                #classifier.train(test_fold)
                #if verbose:
                #    print >> stderr, 'Done!'

                #if verbose:
                #    print >> stderr, 'Evaluating...',
                scores.append(_score_classifier_by_tup(classifier, (test_lbls, test_vecs)))
                #if verbose:
                #    print >> stderr, 'Done!'

                #if len(test_fold) == len(train):
                #    # We only need to sample once when the sample is the
                #    #   same size as what we sample if we are deterministic
                #    break
          
            if verbose:
                print 'Done!'

            macro_scores = [ms for ms, _, _, _, _ in scores]
            micro_scores = [ms for _, ms, _, _, _ in scores]
            tps = [tp for _, _, tp, _, _ in scores]
            fns = [fn for _, _, _, fn, _ in scores]
            res_dics = [d for _, _, _, _, d in scores]

            classifier_result = (
                    _avg(macro_scores), _stddev(macro_scores),
                    _avg(micro_scores), _stddev(micro_scores),
                    _avg(tps), _stddev(tps),
                    _avg(fns), _stddev(fns),
                    res_dics,
                    )
            classifier_results[len(train_fold_lbls)] = classifier_result
            
            '''
            tps = [tp for tp, _, _ in scores]
            fps = [fp for _, fp, _ in scores]
            accs = [acc for _, _, acc in scores]

            classifier_result = (_avg(tps), _avg(fps),
                    _avg(accs), _stddev(accs))
            classifier_results[len(test_fold)] = classifier_result
            '''

            if verbose:
                res_str = ('Results: '
                        'MACRO: {0:.3f} MACRO_STDDEV: {1:.3f} '
                        'MICRO: {2:.3f} MICRO_STDDEV: {3:.3f} '
                        'TP: {4:.3f} FP: {5:.3f}'
                        ).format(*classifier_result)
                print res_str
                '''
                res_str = ('Results: TP: {0:.3f} FP: {1:.3f} '
                        'P: {2:.3f} STDDEV: {3:.3f}'
                        ).format(*classifier_result)
                print >> stderr, res_str
                '''

        results_by_classifier[classifier_id] = classifier_results
    return dataset_id, results_by_classifier
    #results_by_dataset[dataset_id] = results_by_classifier

def _get_quick_pickle_path(outdir):
    return join_path(outdir, 'quick.pickle')

def _quick_test(classifiers, datasets, outdir, verbose=False, worker_pool=None,
        no_simstring_cache=False, use_test_set=False):
    
    if worker_pool is not None:
        raise NotImplementedError

    results_file_path = _get_quick_pickle_path(outdir)
    results_by_dataset = {}

    for dataset_id, dataset_getter in datasets.iteritems():
        if verbose:
            print >> stderr, 'Data set:', dataset_id

        if verbose:
            print >> stderr, 'Caching data set...',
        train, dev, test = dataset_getter()
        if use_test_set:
            train, dev = list(chain(train, dev)), list(test)
        else:
            train, dev = list(train), list(dev)
        if verbose:
            print >> stderr, 'Done!'
      
        if not no_simstring_cache:
            _simstring_caching(classifiers, (train, dev), verbose=verbose)
    
        # Collect the seen type to iterate over later
        seen_types = set()
        results_by_classifier = {}

        for classifier_id, classifier_class in classifiers.iteritems():
            if verbose:
                print >> stderr, 'Classifier:', classifier_id
            
            classifier = classifier_class()

            classifier.train(train)

            score = _score_classifier(classifier, dev)
            results_by_classifier[classifier_id] = score
            macro_score, micro_score, tp_sum, fn_sum, _ = score
            
            if verbose:
                res_str = ('Results: '
                        '{0:.3f}/'
                        '{1:.3f}/'
                        '{2}/{3} (MACRO/MICRO/TP/FN)'
                        ).format(macro_score, micro_score, tp_sum, fn_sum)
                print res_str

        results_by_dataset[dataset_id] = results_by_classifier

    with open(results_file_path, 'wb') as results_file:
        pickle_dump(results_by_dataset, results_file)

    if verbose:
        print >> stderr, 'Results written to:', results_file_path


def _learning_curve_test(classifiers, datasets, outdir,
        verbose=False, no_simstring_cache=False, folds=10, worker_pool=None,
        min_perc=5, max_perc=101, step_perc=5, it_factor=1,
        pickle_name='learning', use_test_set=False
        ):
    ### This part is really generic
    # TODO: We could keep old results... But dangerous, mix-up
    results_file_path = _get_learning_pickle_path(outdir, pickle_name)
    #XXX: RESUME GOES HERE!
    results_by_dataset = {}
    
    # TODO: If we have a single dataset we could do multi for classifiers,
    #       but beware of caching.

    args = [(classifiers, d_id, d_getter, verbose, no_simstring_cache, use_test_set,
            folds, min_perc, max_perc, step_perc, it_factor)
            for d_id, d_getter in datasets.iteritems()]
    '''
    def _learning_curve_test_data_set(classifiers, dataset_id, dataset_getter,
        verbose=False, no_simstring_cache=False, folds=10,
        min_perc=5, max_perc=101, step_perc=5):
    '''

    #XXX: How to solve the keyword args? Order?

    if worker_pool is not None:
        res_it = worker_pool.imap(__learning_curve_test_data_set, args)
    else:
        res_it = (_learning_curve_test_data_set(*arg) for arg in args)

    for dataset_id, dataset_results in res_it:
        results_by_dataset[dataset_id] = dataset_results

        ### HACK TO GET INTERMEDIATE!
        with open(results_file_path, 'w') as results_file:
            pickle_dump(results_by_dataset, results_file)

        if verbose:
            print >> stderr, 'Results written to:', results_file_path
        ###

    with open(results_file_path, 'w') as results_file:
        pickle_dump(results_by_dataset, results_file)

    if verbose:
        print >> stderr, 'Results written to:', results_file_path

def _get_learning_pickle_path(outdir, name='learning'):
    return join_path(outdir, '{0}_results.pickle'.format(name))

# Nice table-able number for a curve
def _learning_curve_avg(classifiers, datasets, outdir, pickle_name='learning'):

    with open(_get_learning_pickle_path(outdir, name=pickle_name), 'r') as results_file:
        results = pickle_load(results_file)

    for dataset in datasets:
        print 'Dataset:', dataset
        for classifier in classifiers:
            print 'Classifier:', classifier
            macro_avg = _avg([res_tup[0] for res_tup
                in results[dataset][classifier].itervalues()])

            print macro_avg

#XXX: This part is MESSY
NO_LEGEND = False
MINIMAL = True
def _plot_learning_curve(outdir, worker_pool=None, pickle_name='learning'):
    # We have to try to import here, or we will crash
    import matplotlib.pyplot as plt

    if worker_pool is not None:
        raise NotImplementedError

    with open(_get_learning_pickle_path(outdir, name=pickle_name), 'r') as results_file:
        results = pickle_load(results_file)

    line_colour_by_classifier = {
            'NAIVE': 'm',
            #'MAXVOTE': 'y',
            'INTERNAL': 'r',
            'SIMSTRING': 'y',
            'INTERNAL-SIMSTRING': 'b',
            #'SIMPLE-INTERNAL-ENSEMBLE': 'g',
            'GAZETTER': 'c',
            'INTERNAL-GAZETTER': 'g',
            #'SIMSTRING-COMPETITIVE': 'm',
            #'COMPETITIVE': 'k',
            } # We need more colours?

    line_style_by_classifier = {
            'NAIVE': '-:', #XXX:
            #'MAXVOTE': 'y',
            'INTERNAL': 'default-.',
            'SIMSTRING': 'steps-pre-.',
            'INTERNAL-SIMSTRING': '-',
            #'SIMPLE-INTERNAL-ENSEMBLE': 'g',
            'GAZETTER': 'c',
            'INTERNAL-GAZETTER': '--',
            #'SIMSTRING-COMPETITIVE': 'm',
            #'COMPETITIVE': 'k',
            }

    plot_dir = outdir
    for dataset, classifiers in results.iteritems():
        fig = plt.figure()
        #plt.title(dataset)
        plt.ylabel('Accuracy')
        plt.xlabel('Training Examples')

        #legendary_dic = {}

        min_seen = 1
        max_seen = 0
        for classifier, classifier_results in classifiers.iteritems():
            if MINIMAL:
                if classifier not in ('INTERNAL', 'INTERNAL-SIMSTRING', 'INTERNAL-GAZETTER', ):
                    continue
                classifier_name = {
                        'INTERNAL': 'Internal',
                        'INTERNAL-SIMSTRING': 'Internal-SimString',
                        'INTERNAL-GAZETTER': 'Internal-Gazetteer',
                        }[classifier]
            else:
                classifier_name = classifier

            res_tups = [(size_value, res_tup[0], res_tup[1], res_tup[2], res_tup[3])
                    for size_value, res_tup in classifier_results.iteritems()]
            res_tups.sort()

            '''
                    _avg(macro_scores), _stddev(macro_scores),
                    _avg(micro_scores), _stddev(micro_scores),
                    _avg(tps), _stddev(tps),
                    _avg(fps), _stddev(fps),
            '''


            sample_sizes = [t[0] for t in res_tups]
            macro_vals = [t[1] for t in res_tups]
            macro_stds = [t[2] for t in res_tups]
            micro_vals = [t[3] for t in res_tups]
            micro_stds = [t[4] for t in res_tups]

            max_seen = max(max_seen, max(macro_vals))
            min_seen = min(max_seen, min(macro_vals))

            #plt.axhline(y=float(i) / 10, color='k')

            #lines, _, _ = 
            plt.errorbar(sample_sizes, macro_vals,
                    #yerr=macro_stds,
                    label=classifier_name,
                    linestyle=line_style_by_classifier[classifier],
                    color='k',
                    #color=line_colour_by_classifier[classifier],
                    )
            #linestyle='--'
            #plt.errorbar(sample_sizes, micro_vals,
            #        #yerr=micro_stds, #label=classifier,
            #        linestyle='--',
            #        color=line_colour_by_classifier[classifier])

            #lines = None
            #legendary_dic[classifier] = lines
            #print >> stderr, legendary_dic
            #line.set_color(line_colour_by_classifier[classifier])

        #ax.legend([c for c, _ in legendary_dic.iteritems()],
        #        [l for _, l in legendary_dic.iteritems()], loc=4)##
        if not NO_LEGEND:
            ax = fig.get_axes()[0]
            handles, labels = ax.get_legend_handles_labels()

            # reverse the order
            ax.legend(handles[::-1], labels[::-1])

            # or sort them by labels
            hl = sorted(zip(handles, labels),
                    key=itemgetter(1))
            handles2, labels2 = zip(*hl)

            ax.legend(handles2, labels2, loc=4)

        '''
        for i in xrange(1, 100):
            val = float(i) / 100
            if min_seen < val < max_seen:
                plt.axhline(y=float(i) / 10, color='k')
        '''
        plt.ylim()#ymax=1.0) #ymin=0.0

        for fmt in ('png', 'svg', ):
            plt.savefig(join_path(plot_dir, dataset.lower() + '_' + pickle_name) + '.' + fmt,
                    format=fmt)

            ### XXX: ST_VIS
ST_VIS_DIR = join_path(dirname(__file__), 'st_vis') 
ST_VIS_FILES = [join_path(ST_VIS_DIR, f)
        for f in ('code.js', 'index.html', 'style.css', )]
ST_VIS_LIB_FILES = [join_path(ST_VIS_DIR, 'lib', f)
        for f in listdir(join_path(ST_VIS_DIR, 'lib'))
        if f.endswith('.js') or f.endswith('.gif')]

def _set_up_st_vis_dir(st_vis_dir):
    for file_path in ST_VIS_FILES:
        copy(file_path, join_path(st_vis_dir, basename(file_path)))

    # TODO: Should this check be here? Overwrite protection?
    try:
        makedirs(join_path(outdir, 'lib'))
    except OSError, e:
        if e.errno == 17:
            pass
        else:
            raise

    for file_path in ST_VIS_LIB_FILES:
        copy(file_path, join_path(outdir, 'lib', basename(file_path)))
###

def _simstring_caching(classifiers, document_sets, verbose=False):
    if any((True for k in classifiers
        # NOTE: Keep this check up to date
        if 'SIMSTRING' in k or 'GAZETTER' in k or 'TSURUOKA' in k)):
        if verbose:
            print >> stderr, 'Caching queries for SimString:'

        _cache_simstring(document_sets, verbose=verbose)
    else:
        if verbose:
            print >> stderr, 'No caching necessary for the given classifier'

def _confusion_matrix_test(classifiers, datasets, outdir,
        verbose=False, no_simstring_cache=False, worker_pool=None):
    results_by_dataset = {}

    if worker_pool is not None:
        raise NotImplementedError

    for dataset_id, dataset_getter in datasets.iteritems():
        if verbose:
            print >> stderr, 'Data set:', dataset_id

        if verbose:
            print >> stderr, 'Caching data set...',
        train, dev, _ = dataset_getter()
        train, dev = list(train), list(dev)
        if verbose:
            print >> stderr, 'Done!'

        if not no_simstring_cache:
            _simstring_caching(classifiers, (train, dev), verbose=verbose)

        # Collect the seen type to iterate over later
        seen_types = set()
        results_by_classifier = {}

        for classifier_id, classifier_class in classifiers.iteritems():
            if verbose:
                print >> stderr, 'Classifier:', classifier_id

            classifier = classifier_class()

            if verbose:
                print >> stderr, 'Training...',
            classifier.train(train)
            if verbose:
                print >> stderr, 'Done!'

            confusion_matrix = defaultdict(lambda : defaultdict(int))
            if verbose:
                print >> stderr, 'Evaluating...',
            for document in dev:
                for sentence in document:
                    for annotation in sentence:
                        predicted = classifier.classify(document,
                                sentence, annotation)

                        #if predicted != annotation.type:
                        confusion_matrix[annotation.type][predicted] += 1

                        seen_types.add(annotation.type)
            if verbose:
                print >> stderr, 'Done!'

            results_by_classifier[classifier_id] = confusion_matrix

        for _, confusion_matrix in results_by_classifier.iteritems():
            for from_type in confusion_matrix:
                for type in seen_types:
                    if type not in confusion_matrix:
                        confusion_matrix[type] = 0
                # Set the already existing values to the same and other unseen
                #       but existing to 0
                #XXX: HACK!
                #if type not in confusion_matrix:
                #    confusion_matrix[type] = 0
                #confusion_matrix[type] = confusion_matrix[type]

        results_by_dataset[dataset_id] = results_by_classifier

    json_dic = defaultdict(lambda : defaultdict(dict))
    for dataset_id, results_by_classifier in results_by_dataset.iteritems():
        for classifier_id, confusion_matrix in results_by_classifier.iteritems():
            # Get the column names and order
            col_names = [k for k in confusion_matrix]
            col_names.sort()

            max_by_col = defaultdict(int)
            min_by_col = defaultdict(lambda : 2**32)

            table_data = []
            for row_name in col_names:
                row_data = confusion_matrix[row_name]

                # Not yet normalised
                raw_row = [row_data[k] for k in col_names]
                row_sum = float(sum(raw_row))

                table_data.append([e / row_sum for e in raw_row])

                for k in col_names:
                    max_by_col[k] = max(row_data[k], max_by_col[k])
                    min_by_col[k] = min(row_data[k], min_by_col[k])

            col_ranges = [None for _ in col_names]
            col_frmts = ['%.2f' for _ in col_names]

            # Now append a name to each row
            col_ranges.insert(0, None)
            col_frmts.insert(0, '%s')

            for row_name, row in izip(col_names, table_data):
                row.insert(0, row_name)

            col_names.insert(0, 'From \\ To')

            table_ftr = [-1 for _ in col_names]

            # NAMES, FRMT, RANGES, TABLE_DATA, TABLE_FTR
            json_dic[classifier_id][dataset_id] = (
                    col_names,
                    col_frmts,
                    col_ranges,
                    table_data,
                    table_ftr,
                    )

            # XXX: Will overwrite!
    with open(join_path(outdir, 'data.js'), 'w') as data_file:
        data_file.write('data = ')
        data_file.write(json_dumps(json_dic, encoding='utf-8',
            indent=2, sort_keys=True))

        if verbose:
            print >> stderr, 'Output: file://{0}'.format(
                    abspath(join_path(outdir, 'index.html')))

def _censor_sparse_vectors_gen(vecs, idxs):
    for vec in vecs:
        new_vec = hashabledict()
        for idx in vec:
            if idx not in idxs:
                new_vec[idx] = vec[idx]
        yield new_vec

def _lexical_descent(classifiers, datasets, outdir, verbose=False,
        worker_pool=None, no_simstring_cache=False, use_test_set=False):
    # Check that we can in fact do a lexical descent for the classifier
    for classifier_name in classifiers:
        assert ('SIMSTRING' in classifier_name
                or 'TSURUOKA' in classifier_name
                or 'GAZETTER' in classifier_name)

    for classifier_name, classifier_class in classifiers.iteritems():
        print 'Classifier:', classifier_name
        classifier =  classifier_class()

        for dataset_name, dataset_getter in datasets.iteritems():
            print 'Dataset:', dataset_name
            if verbose:
                print >> stderr, 'Reading data...',

            train_set, dev_set, test_set = dataset_getter()
            if use_test_set:
                train, test = list(chain(train_set, dev_set)), list(test_set)
            else:
                train, test = list(train_set), list(dev_set)
            del train_set, dev_set, test_set

            if verbose:
                print >> stderr, 'Done!'

            if not no_simstring_cache:
                _simstring_caching((classifier_name, ),
                    (train, test, ), verbose=verbose)


            train_lbls, train_vecs = classifier._gen_lbls_vecs(train)
            test_lbls, test_vecs = classifier._gen_lbls_vecs(test)
            train_vecs = [hashabledict(d) for d in train_vecs]
            test_vecs = [hashabledict(d) for d in test_vecs]
            train_uncensored_vecs = deepcopy(train_vecs)

            # Generate the folds for all iterations
            folds = [f for f in _k_folds(5,
                set(izip(train_lbls, train_vecs)))] #XXX: Constant

            # XXX: This is an ugly hack and bound to break:
            # Locate which vector ID;s that are used by SimString features and
            # by which feature
            from classifier.simstring.features import SIMSTRING_FEATURES
            sf_ids = [f().get_id() for f in SIMSTRING_FEATURES]

            vec_idxs_by_feat_id = defaultdict(set)
            for sf_id in sf_ids:
                for f_id in classifier.vec_index_by_feature_id:
                    # NOTE: Not 100% safe check, could match by accident
                    if sf_id in f_id:
                        vec_idxs_by_feat_id[sf_id].add(
                                classifier.vec_index_by_feature_id[f_id])

            # Which ones never fired?
            i = 0
            for i, sf_id in enumerate((id for id in sf_ids
                if id not in vec_idxs_by_feat_id), start=1):
                print sf_id, 'never fired'
            else:
                print '{} SimString feature(s) never fired'.format(i)

            res_dic = defaultdict(lambda : defaultdict(lambda : '-'))

            # Iteratively find the best candidate
            to_evaluate = set((f_id for f_id in vec_idxs_by_feat_id))
            removed = set()
            iteration = 1
            last_macro_score = None
            while to_evaluate:
                print 'Iteration:', iteration

                print 'Censoring vectors...',
                # Censor everything we have removed so far
                idxs_to_censor = set(i for i in chain(
                    *(vec_idxs_by_feat_id[f_id] for f_id in removed)))
                train_vecs = [d for d in _censor_sparse_vectors_gen(
                    train_vecs, idxs_to_censor)]

                train_data = set(izip(train_lbls, train_vecs))

                train_folds = []
                for fold in folds:
                    f_lbls = (l for l, _ in fold)
                    f_vecs = (d for d in _censor_sparse_vectors_gen(
                        (v for _, v in fold), idxs_to_censor))
                    train_folds.append(set(izip(f_lbls, f_vecs)))
                print 'Done!'
                
                print 'Training and evaluating a model of our current state...',
                classifier._liblinear_train(train_lbls, train_vecs)
                print 'Done!'

                test_censored_vecs = [d for d in _censor_sparse_vectors_gen(
                    test_vecs, idxs_to_censor)]
                curr_macro_score = _score_classifier_by_tup(classifier,
                        (test_lbls, test_censored_vecs))[0]

                print 'Current state on test is: {}'.format(curr_macro_score)
                if last_macro_score is not None:
                    print 'Last state was: {} (diff: {})'.format(last_macro_score,
                        curr_macro_score - last_macro_score)
                last_macro_score = curr_macro_score

                # Prepare to go parallel
                f_args = ((f_id, classifier, train_data, train_folds,
                    to_censor) for f_id, to_censor
                    in vec_idxs_by_feat_id.iteritems() if f_id in to_evaluate)
                # Also cram in our non-censored one in there
                f_args = chain(((None, classifier, train_data, train_folds,
                    set()), ), f_args)

                score_by_knockout = {}
                print 'Evaluating knockouts ({} in total)'.format(
                        len(to_evaluate) + 1)
                # TODO: A bit reduntant, prettify!
                if worker_pool is not None:
                    i = 1
                    for f_id, mean in worker_pool.imap_unordered(
                            __knockout_pass, f_args):
                        score_by_knockout[f_id] = mean
                        print 'it: {} k: {} res: {} {}'.format(
                                iteration, i, f_id, mean)
                        i += 1
                else:
                    for i, args in enumerate(f_args, start=1):
                        f_id, mean = _knockout_pass(*args)
                        score_by_knockout[f_id] = mean
                        print 'it: {} k: {} res: {} {}'.format(
                                iteration, i, f_id, mean)

                # Set the result dictionary
                for f_id, mean in score_by_knockout.iteritems():
                    res_dic[str(iteration)][f_id] = mean
                # And write the results incrementally for each round
                with open(join_path(outdir, 'descent_{}_{}.md'.format(
                    classifier_name, dataset_name)), 'w') as md_file:
                    from md import dict_to_table
                    md_file.write(dict_to_table(res_dic, total=False, perc=False))
                    md_file.write('\n')
                
                # Find the best scoring one...
                scores = [(s, f_id)
                        for f_id, s in score_by_knockout.iteritems()]
                scores.sort()
                scores.reverse()

                best_score, best_f_id = scores[0]

                print 'Round winner: {} with {}'.format(best_f_id, best_score)

                if best_f_id is None:
                    # We are done, no removal gave a better score
                    break

                removed.add(best_f_id)
                to_evaluate.remove(best_f_id)
                
                iteration += 1

            if removed:
                # TODO: Could do more metrics here?

                print 'Training and evaluating a model of our previous state...',
                classifier._liblinear_train(train_lbls, train_uncensored_vecs)
                before_macro_score = _score_classifier_by_tup(classifier,
                        (test_lbls, test_vecs))[0]
                print 'Done!'

                print 'Training and evaluating a model of our current state...',
                train_censored_vecs = [d for d in _censor_sparse_vectors_gen(train_vecs,
                    set(i for i in chain(*(vec_idxs_by_feat_id[f_id] for f_id in removed))))]
                classifier._liblinear_train(train_lbls, train_censored_vecs)
                print 'Done!'

                test_censored_vecs = [d for d in _censor_sparse_vectors_gen(test_vecs,
                    set(i for i in chain(*(vec_idxs_by_feat_id[f_id] for f_id in removed))))]
                after_macro_score = _score_classifier_by_tup(classifier,
                        (test_lbls, test_censored_vecs))[0]

                res_str = 'Before: {} After: {}'.format(before_macro_score,
                        after_macro_score)
                print res_str
                print 'Happy?'
            else:
                res_str = 'Unable to remove any lexical resource to make improvements...'
                print res_str

            # Ugly but saves the final result safely
            with open(join_path(outdir, 'descent_{}_{}.txt'.format(
                classifier_name, dataset_name)), 'w') as res_file:
                res_file.write(res_str)

def __knockout_pass(args):
    return _knockout_pass(*args)

def _knockout_pass(f_id, classifier, train_data, folds, to_censor):
    macro_scores = []
    for fold_num, fold in enumerate(folds, start=1):
        train_set = train_data - fold
        test_set = fold

        assert len(train_set) + len(test_set) == len(train_data)

        train_vecs = [d for d in _censor_sparse_vectors_gen(
            (v for _, v in train_set), to_censor)]
        train_lbls = [l for l, _ in train_set]

        classifier._liblinear_train(train_lbls, train_vecs)

        test_vecs = [d for d in _censor_sparse_vectors_gen(
            (v for _, v in test_set), to_censor)]
        test_lbls = (l for l, _ in test_set)
        res_tup =_score_classifier_by_tup(classifier, (test_lbls, test_vecs))
        macro_scores.append(res_tup[0])

    mean = _mean(macro_scores)

    return f_id, mean

def _cache(datasets, verbose=False):
    dataset_getters = []
    for dataset_name, dataset_getter in datasets.iteritems():
        if verbose:
            print >> stderr, 'Reading data for {}...'.format(dataset_name),
        for getter in dataset_getter():
            dataset_getters.append(getter)
        if verbose:
            print >> stderr, 'Done!'

    _simstring_caching(('SIMSTRING', ),
        dataset_getters, verbose=verbose)

def main(args):
    argp = ARGPARSER.parse_args(args[1:])

    # Shuffle around the arguments and turn them into dicts
    datasets = [d for d in argp.dataset]
    datasets = OrderedDict([(d, DATASETS[d]) for d in datasets])

    classifiers = [c for c in argp.classifier]
    classifiers = OrderedDict([(c, CLASSIFIERS[c]) for c in classifiers])

    outdir = argp.outdir
    tests = argp.test
    verbose = argp.verbose
    # TODO: Rename no_simstring_cache into no_pre_cache
    no_simstring_cache = argp.no_pre_cache
    if argp.jobs > 1:
        worker_pool = Pool(argp.jobs)
    else:
        worker_pool = None

    # Delegate each test
    #TODO: Parallel tests?
    for test in chain(*tests):
        if test == 'confusion':
            print >> stderr, 'WARNING: Old unsupported code'
            _confusion_matrix_test(classifiers, datasets, outdir,
                    verbose=verbose, no_simstring_cache=no_simstring_cache)
        elif test == 'learning':
            _learning_curve_test(classifiers, datasets, outdir,
                    verbose=verbose, no_simstring_cache=no_simstring_cache,
                    worker_pool=worker_pool, use_test_set=argp.test_set)
        elif test == 'low-learning':
            _learning_curve_test(classifiers, datasets, outdir,
                    verbose=verbose, no_simstring_cache=no_simstring_cache,
                    worker_pool=worker_pool, it_factor=None,
                    min_perc=1, max_perc=30, step_perc=2,
                    pickle_name='low_learning', use_test_set=argp.test_set
                    )
        elif test == 'plot':
            _plot_learning_curve(outdir)
        elif test == 'low-plot':
            _plot_learning_curve(outdir, pickle_name='low_learning')
        elif test == 'quick':
            _quick_test(classifiers, datasets, outdir,
                    verbose=verbose, no_simstring_cache=no_simstring_cache,
                    worker_pool=worker_pool, use_test_set=argp.test_set)
        elif test == 'cache':
            _cache(datasets, verbose=verbose)
        elif test == 'learning-avg':
            _learning_curve_avg(classifiers, datasets, outdir)
        elif test == 'lex-descent':
            _lexical_descent(classifiers, datasets, outdir,
                verbose=verbose, no_simstring_cache=no_simstring_cache,
                worker_pool=worker_pool, use_test_set=argp.test_set)
        else:
            assert False, 'Unimplemented test case, {}'.format(test)

    return 0

### Trailing Constants
ARGPARSER = ArgumentParser(description='SimSem test-suite')
ARGPARSER.add_argument('outdir', type=writeable_dir)
# Test commands
ARGPARSER.add_argument('test', choices=('confusion', 'learning', 'plot',
    'quick', 'low-learning', 'low-plot', 'cache', 'learning-avg',
    'lex-descent', ),
    action='append', nargs='+')
ARGPARSER.add_argument('-c', '--classifier', default=[],
        choices=tuple([c for c in CLASSIFIERS]),
        help='classifier(s) to use for the test(s)', action='append')
ARGPARSER.add_argument('-d', '--dataset', default=[],
        choices=tuple([d for d in DATASETS]),
        help='dataset(s) to use for the test(s)', action='append')
ARGPARSER.add_argument('-n', '--no-pre-cache', action='store_true',
        help=('disables pre-caching, gives some performance increase if this '
            'has been done previously'))
ARGPARSER.add_argument('-j', '--jobs', type=int, default=1)
ARGPARSER.add_argument('-v', '--verbose', action='store_true')
ARGPARSER.add_argument('-X', '--test-set', action='store_true',
        help=('use the test set for evaluation and train and development for '
            'training, ONLY use this for final result generation'))
###

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
