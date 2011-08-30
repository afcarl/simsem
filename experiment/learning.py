'''
Learning-curve test functionality.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-29
'''

from itertools import chain
from operator import itemgetter
from os.path import join as path_join
from random import sample
from sys import stderr

from common import compress, simstring_caching
from maths import mean, stddev
from scoring import score_classifier_by_tup, score_classifier_by_tup_ranked

try:
    from cPickle import dump as pickle_dump, load as pickle_load
except ImportError:
    from pickle import dump as pickle_dump, load as pickle_load

# multiprocessing argument splitting
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
    if verbose:
        print >> stderr, 'Generating filters...',
    train_filters = []
    indices = [i for i in xrange(train_size)]
    for p in xrange(min_perc, max_perc, step_perc):
        sample_size = int((p / 100.0) * train_size)
        # XXX: Heuristic, * 2?

        if it_factor is not None and False:
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
        simstring_caching(classifiers, (train, dev), verbose=verbose)

    # Collect the seen type to iterate over later
    seen_types = set()
    results_by_classifier = {}

    for classifier_id, classifier_class in classifiers.iteritems():
        if verbose:
            print >> stderr, 'Classifier:', classifier_id,
            
        classifier = classifier_class()
        train_lbls, train_vecs = classifier._gen_lbls_vecs(train)
        test_lbls,  test_vecs = classifier._gen_lbls_vecs(dev)

        assert len(train_lbls) == train_size, '{} != {}'.format(
                len(train_lbls), train_size)

        classifier_results = {}
        
        for train_fold_filters in train_filters:
            
            scores = []
            new_scores = []
            for i, train_fold_filter in enumerate(train_fold_filters, start=1):
                if verbose and i % 10 == 0:
                    print >> stderr, i, '...',

                train_fold_lbls = [l for l in compress(train_lbls, train_fold_filter)]
                train_fold_vecs = [v for v in compress(train_vecs, train_fold_filter)]

                assert train_fold_lbls, train_fold_filter
                assert train_fold_vecs, train_fold_filter
                classifier._train(train_fold_lbls, train_fold_vecs)

                scores.append(score_classifier_by_tup(classifier,
                    (test_lbls, test_vecs)))
                # XXX: Hooking new scores into the old learning
                new_scores.append(score_classifier_by_tup_ranked(classifier,
                    (test_lbls, test_vecs), unseen=True))
          
            if verbose:
                print 'Done!'

            macro_scores = [ms for ms, _, _, _, _ in scores]
            micro_scores = [ms for _, ms, _, _, _ in scores]
            tps = [tp for _, _, tp, _, _ in scores]
            fns = [fn for _, _, _, fn, _ in scores]
            res_dics = [d for _, _, _, _, d in scores]

            # New metrics
            ranks = [r for r in chain(*(rs for rs, _, _ in new_scores))] 
            ambiguities = [a for a in chain(*(ambs for _, ambs, _ in new_scores))]
            losses = [loss for  _, _, loss in new_scores]

            ranks_mean = mean(ranks)
            ranks_stddev = stddev(ranks)
            ambiguities_mean = mean(ambiguities)
            ambiguities_stddev = stddev(ambiguities)
            losses_mean = mean(losses)
            losses_stddev = stddev(losses)

            classifier_result = (
                    mean(macro_scores), stddev(macro_scores),
                    mean(micro_scores), stddev(micro_scores),
                    mean(tps), stddev(tps),
                    mean(fns), stddev(fns),
                    res_dics,
                    # New metrics
                    ranks_mean, ranks_stddev,
                    ambiguities_mean, ambiguities_stddev,
                    losses_mean, losses_stddev,
                    )


            classifier_results[len(train_fold_lbls)] = classifier_result
            
            if verbose:
                res_str = ('Results {size}: '
                        'MACRO: {0:.3f} MACRO_STDDEV: {1:.3f} '
                        'MICRO: {2:.3f} MICRO_STDDEV: {3:.3f} '
                        'TP: {4:.3f} FP: {5:.3f} '
                        'MEAN_RANK: {mean_rank:.3f} MEAN_RANK_STDDEV: {mean_rank_stddev:.3f} '
                        'AVG_AMB: {avg_amb:.3f} AVG_AMB_STDDEV: {avg_amb_stddev:.3f} '
                        'LOST: {lost:.3f} LOST_STDDEV: {lost_stddev:.3f}'
                        ).format(*classifier_result,
                                size=len(train_fold_lbls),
                                mean_rank=ranks_mean,
                                mean_rank_stddev=ranks_stddev,
                                avg_amb=ambiguities_mean,
                                avg_amb_stddev=ambiguities_stddev,
                                lost=losses_mean,
                                lost_stddev=losses_stddev
                                )
                print res_str

        results_by_classifier[classifier_id] = classifier_results
    return dataset_id, results_by_classifier

def learning_curve_test(classifiers, datasets, outdir,
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
    return path_join(outdir, '{0}_results.pickle'.format(name))

# Nice table-able number for a curve
def learning_curve_avg(classifiers, datasets, outdir, pickle_name='learning'):

    with open(_get_learning_pickle_path(outdir, name=pickle_name), 'r') as results_file:
        results = pickle_load(results_file)

    for dataset in datasets:
        print 'Dataset:', dataset
        for classifier in classifiers:
            print 'Classifier:', classifier
            macro_avg = mean([res_tup[0] for res_tup
                in results[dataset][classifier].itervalues()])

            print macro_avg

def _plot_curve(plot_dir, results, plot_name, new_metric=False):
    import matplotlib.pyplot as plt

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

    for dataset, classifiers in results.iteritems():
        fig = plt.figure()
        if not new_metric:
            plt.ylabel('Accuracy')
            plt.xlabel('Training Examples')
        else:
            plt.ylabel('Ambiguity')
            plt.xlabel('Training Examples')

        min_seen = 1
        max_seen = 0
        for classifier, classifier_results in classifiers.iteritems():
            if classifier != 'INTERNAL-SIMSTRING': #XXX:
                continue
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

            # TODO: Get rid of all this index sillines, named tuple...
            res_tups = [(size_value, res_tup[0], res_tup[1], res_tup[2],
                res_tup[3], res_tup[4], res_tup[5], res_tup[6])
                    for size_value, res_tup in classifier_results.iteritems()]
            res_tups.sort()

            sample_sizes = [t[0] for t in res_tups]
            macro_vals = [t[1] for t in res_tups]
            macro_stds = [t[2] for t in res_tups]
            micro_vals = [t[3] for t in res_tups]
            micro_stds = [t[4] for t in res_tups]
            # New metrics
            rank_mean = [t[5] for t in res_tups]
            avg_ambiguity_sizes = [t[6] for t in res_tups]
            losses_by_threshold = [t[7] for t in res_tups]

            max_seen = max(max_seen, max(macro_vals))
            min_seen = min(max_seen, min(macro_vals))

            if not new_metric:
                plt.errorbar(sample_sizes, macro_vals,
                        yerr=macro_stds,
                        label=classifier_name,
                        linestyle=line_style_by_classifier[classifier],
                        color='k',
                        # Disabled colour plotting
                        #color=line_colour_by_classifier[classifier],
                        )
            else:
                plt.errorbar(sample_sizes, avg_ambiguity_sizes,
                        label=classifier_name,
                        color='k',
                        )
        if not NO_LEGEND:
            if not new_metric:
                ax = fig.get_axes()[0]
                handles, labels = ax.get_legend_handles_labels()

                # reverse the order
                ax.legend(handles[::-1], labels[::-1])

                # or sort them by labels
                hl = sorted(zip(handles, labels),
                        key=itemgetter(1))
                handles2, labels2 = zip(*hl)

                ax.legend(handles2, labels2, loc=4)
            else:
                pass #XXX

        plt.ylim()#ymax=1.0) #ymin=0.0

        for fmt in ('png', 'svg', ):
            plt.savefig(path_join(plot_dir, dataset.lower() + '_' + plot_name) + '.' + fmt,
                    format=fmt)

    pass

#XXX: This part is MESSY
NO_LEGEND = False
MINIMAL = True
def plot_learning_curve(outdir, worker_pool=None, pickle_name='learning'):
    # We have to try to import here, or we will crash

    if worker_pool is not None:
        raise NotImplementedError

    with open(_get_learning_pickle_path(outdir, name=pickle_name), 'r'
            ) as results_file:
        results = pickle_load(results_file)

    _plot_curve(outdir, results, pickle_name)
    _plot_curve(outdir, results, pickle_name + '_ambiguity', new_metric=True)
