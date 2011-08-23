#!/usr/bin/env python

'''
Experimental test-suite for SimSem.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-03-02
'''

from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from itertools import izip, chain, tee
from json import dumps as json_dumps
from math import sqrt
from multiprocessing import Pool
from os import listdir
from os import makedirs
from os.path import basename, abspath, dirname
from os.path import join as join_path
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from random import sample, seed, random
from shutil import copy
from sys import path as sys_path
from sys import stderr

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

from itertools import izip

# I hate Python 2.6...
def _compress(it, flter):
    for e, v in izip(it, flter):
        if v:
            yield e

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

from itertools import izip

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
    #from itertools import compress
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
            import operator
            hl = sorted(zip(handles, labels),
                    key=operator.itemgetter(1))
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
    for test in tests:
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
        else:
            assert False, 'Unimplemented test case'

    return 0

### Trailing Constants
ARGPARSER = ArgumentParser(description='SimSem test-suite')
ARGPARSER.add_argument('outdir', type=writeable_dir)
# Test commands
ARGPARSER.add_argument('test', choices=('confusion', 'learning', 'plot',
    'quick', 'low-learning', 'low-plot', 'cache', 'learning-avg', ),
    action='append')
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
