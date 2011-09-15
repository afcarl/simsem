'''
Learning-curve test functionality.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-29
'''

from collections import defaultdict
from itertools import chain, izip
from operator import itemgetter
from os.path import join as path_join
from random import sample, seed
from sys import stderr

from common import compress, simstring_caching
from maths import mean, stddev
from scoring import score_classifier_by_tup, score_classifier_by_tup_ranked

try:
    from cPickle import dump as pickle_dump, load as pickle_load
except ImportError:
    from pickle import dump as pickle_dump, load as pickle_load

def __train_fold(args):
    return _train_fold(*args)

def _train_fold(classifier, train_fold):
    train_fold_lbls = [lbl for lbl, _ in train_fold]
    train_fold_vecs = [vec for _, vec in train_fold]
    assert len(train_fold_lbls) == len(train_fold_vecs)

    classifier._train(train_fold_lbls, train_fold_vecs)
    return len(train_fold_vecs), classifier

def _score_classifier(classifier, test_lbls, test_vecs):
    score = score_classifier_by_tup(classifier,
        (test_lbls, test_vecs))
    # XXX: Hooking new scores into the old learning
    new_score = score_classifier_by_tup_ranked(classifier,
        (test_lbls, test_vecs), unseen=True)
    return score, new_score

def _train_fold_gen(data_set, min_perc, max_perc, step_perc, it_factor):
    set_size = len(data_set)
    # Start with the largest folds, they take longer to process
    for p in xrange(max_perc, min_perc - 1, -step_perc):
        # Sample size for this iteration
        sample_size = int((p / 100.0) * set_size)

        if it_factor is not None:
            folds = int(int(set_size / float(sample_size)) * it_factor)
        else:
            folds = 1
    
        if p == 100:
            # We can't sample when we use the whole set...
            folds = 1
        # Heuristic to keep us from having too low of a sample
        elif folds < 4:
            folds = 4

        for _ in xrange(folds * 2):
            yield sample(data_set, sample_size)

def _learning_curve_test_data_set(classifiers, train, test,
        worker_pool, verbose=False, no_simstring_cache=False,
        use_test_set=False, folds=10, min_perc=5, max_perc=100, step_perc=5,
        it_factor=1):

    # XXX: Not necessary any more!
    if verbose:
        print >> stderr, 'Calculating train set size...',
    train_size = 0
    for d in train:
        for s in d:
            for a in s:
                train_size += 1
    if verbose:
        print >> stderr, 'Done!'
    # XXX:

    if not no_simstring_cache:
        simstring_caching(classifiers, (train, test), verbose=verbose)

    # Collect the seen type to iterate over later
    seen_types = set()
    results_by_classifier = {}

    for classifier_id, classifier_class in classifiers.iteritems():
        if verbose:
            print >> stderr, 'Classifier:', classifier_id, '...',

        from classifier.liblinear import hashabledict

        classifier = classifier_class()
        if verbose:
            print >> stderr, 'featurising train:', '...',
        train_lbls, train_vecs = classifier._gen_lbls_vecs(train)
        train_set = [e for e in izip(train_lbls, train_vecs)]
        assert len(train_lbls) == train_size, '{} != {}'.format(
                len(train_lbls), train_size)
        assert len(train_vecs) == train_size, '{} != {}'.format(
                len(train_vecs), train_size)
        assert len(train_set) == train_size, '{} != {}'.format(
                len(train_set), train_size)
        del train_lbls
        del train_vecs
        if verbose:
            print >> stderr, 'Done!',
            print >> stderr, 'featurising test', '...',
        test_lbls, test_vecs = classifier._gen_lbls_vecs(test)
        test_vecs = [hashabledict(d) for d in test_vecs]
        if verbose:
            print >> stderr, 'Done!',

        # Fix the seed so that we get comparable folds
        seed(0xd5347d33)
        args = ((classifier, fold) for fold in _train_fold_gen(train_set,
            min_perc, max_perc, step_perc, it_factor))

        if worker_pool is None:
            res_it = (_train_fold(*arg) for arg in args)
        else:
            res_it = worker_pool.imap(__train_fold, args)

        classifier_results = defaultdict(list)

        print >> stderr, 'Training and evaluating models: ...',

        i = 0
        for sample_size, fold_classifier in res_it:
            score, new_score = _score_classifier(fold_classifier, test_lbls,
                    test_vecs)
            classifier_results[sample_size].append((score, new_score))
            i += 1
            if i % 10 == 0:
                print >> stderr, i, '...',
        print >> stderr, 'Done!'

        # Process the results
        for sample_size in sorted(e for e in classifier_results):
            results = classifier_results[sample_size]
            scores = [score for score, _ in results]
            new_scores = [new_score for _, new_score in results]

            macro_scores = [ms for ms, _, _, _, _ in scores]
            micro_scores = [ms for _, ms, _, _, _ in scores]
            tps = [tp for _, _, tp, _, _ in scores]
            fns = [fn for _, _, _, fn, _ in scores]
            res_dics = [d for _, _, _, _, d in scores]

            # New metrics
            ranks = [mean(rs) for rs, _, _ in new_scores]
            ambiguities = [mean(ambs) for _, ambs, _ in new_scores]
            recalls = [r for  _, _, r in new_scores]

            # These are means of means
            ranks_mean = mean(ranks)
            ranks_stddev = stddev(ranks)
            ambiguities_mean = mean(ambiguities)
            ambiguities_stddev = stddev(ambiguities)
            recalls_mean = mean(recalls)
            recalls_stddev = stddev(recalls)

            classifier_result = (
                    mean(macro_scores), stddev(macro_scores),
                    mean(micro_scores), stddev(micro_scores),
                    mean(tps), stddev(tps),
                    mean(fns), stddev(fns),
                    res_dics,
                    # New metrics
                    ranks_mean, ranks_stddev,
                    ambiguities_mean, ambiguities_stddev,
                    recalls_mean, recalls_stddev
                    )


            classifier_results[sample_size] = classifier_result
            
            if verbose:
                res_str = ('Results {size}: '
                        'MACRO: {0:.3f} MACRO_STDDEV: {1:.3f} '
                        'MICRO: {2:.3f} MICRO_STDDEV: {3:.3f} '
                        'TP: {4:.3f} FP: {5:.3f} '
                        'MEAN_RANK: {mean_rank:.3f} MEAN_RANK_STDDEV: {mean_rank_stddev:.3f} '
                        'AVG_AMB: {avg_amb:.3f} AVG_AMB_STDDEV: {avg_amb_stddev:.3f} '
                        'RECALL: {recall:.3f} RECALL_STDDEV: {recall_stddev:.3f}'
                        ).format(*classifier_result,
                                size=sample_size,
                                mean_rank=ranks_mean,
                                mean_rank_stddev=ranks_stddev,
                                avg_amb=ambiguities_mean,
                                avg_amb_stddev=ambiguities_stddev,
                                recall=recalls_mean,
                                recall_stddev=recalls_stddev
                                )
                print res_str

        results_by_classifier[classifier_id] = classifier_results
    return results_by_classifier

def learning_curve_test(classifiers, datasets, outdir,
        verbose=False, no_simstring_cache=False, folds=10, worker_pool=None,
        min_perc=5, max_perc=100, step_perc=5, it_factor=1,
        pickle_name='learning', use_test_set=False
        ):
    ### This part is really generic
    # TODO: We could keep old results... But dangerous, mix-up
    results_file_path = _get_learning_pickle_path(outdir, pickle_name)
    #XXX: RESUME GOES HERE!
    results_by_dataset = {}
    
    for dataset_id, dataset_getter in datasets.iteritems():
        
        if verbose:
            print >> stderr, 'Data set:', dataset_id
            
        if verbose:
            print >> stderr, 'Caching vectorised data...',

        train_set, dev_set, test_set = dataset_getter()
        if use_test_set:
            train, test = list(chain(train_set, dev_set)), list(test_set)
        else:
            train, test = list(train_set), list(dev_set)
        del train_set, dev_set, test_set

        if verbose:
            print >> stderr, 'Done!'

        results_by_dataset[dataset_id] = _learning_curve_test_data_set(
                classifiers, train, test, worker_pool,
                verbose=verbose, no_simstring_cache=no_simstring_cache,
                use_test_set=use_test_set, folds=folds, min_perc=min_perc,
                max_perc=max_perc, step_perc=step_perc, it_factor=it_factor)

        ### HACK TO GET INTERMEDIATE!
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
                in results[dataset][classifier].itervalues()]) * 100
            macro_tip = sorted((size, res_tup[0]) for size, res_tup
                    in results[dataset][classifier].iteritems())[-1][1] * 100
            amb_avg = mean([res_tup[11] for res_tup
                in results[dataset][classifier].itervalues()])
            amb_tip = sorted((size, res_tup[11]) for size, res_tup
                    in results[dataset][classifier].iteritems())[-1][1]
            rec_avg = mean([res_tup[13] for res_tup
                in results[dataset][classifier].itervalues()]) * 100
            rec_tip = sorted((size, res_tup[13]) for size, res_tup
                    in results[dataset][classifier].iteritems())[-1][1] * 100

            print ('{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f} '
                    'MACROAVG/MACROTIP/AMBAVG/AMBTIP/RECAVG/RECTIP').format(
                    macro_avg, macro_tip, amb_avg, amb_tip, rec_avg, rec_tip)

### Plot constants
# Default is black
LINE_COLOUR_BY_CLASSIFIER = defaultdict(lambda : 'k')
LINE_COLOUR_BY_CLASSIFIER.update({
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
        })
# NOTE: Turning them all black
for k in LINE_COLOUR_BY_CLASSIFIER:
    LINE_COLOUR_BY_CLASSIFIER[k] = 'k'
LINE_COLOUR_BY_DATASET = defaultdict(lambda : 'k')

LINE_STYLE_BY_CLASSIFIER = defaultdict(lambda : '-')
LINE_STYLE_BY_CLASSIFIER.update({
        'NAIVE': '-:',
        #'MAXVOTE': 'y',
        'INTERNAL': 'default-.',
        'SIMSTRING': 'steps-pre-.',
        'INTERNAL-SIMSTRING': '-',
        #'SIMPLE-INTERNAL-ENSEMBLE': 'g',
        'GAZETTER': 'c',
        'INTERNAL-GAZETTER': '--',
        #'SIMSTRING-COMPETITIVE': 'm',
        #'COMPETITIVE': 'k',
        })

LINE_STYLE_BY_DATASET = defaultdict(lambda : '-')

LINE_MARKER_BY_CLASSIFIER = defaultdict(lambda : 'None')
LINE_MARKER_BY_CLASSIFIER.update({
    })

LINE_MARKER_BY_DATASET = defaultdict(lambda : 'None')
LINE_MARKER_BY_DATASET.update({
    'BioNLP-ST-2011-Epi_and_PTM': 'o',
    'BioNLP-ST-2011-Infectious_Diseases': '|',
    'BioNLP-ST-2011-genia': 's',
    'CALBC_CII': '*',
    'NLPBA': '^',
    'SUPER_GREC': 'x',
    })

# Res-tup handling (yuck):
#res_tups = [(size_value, res_tup[0], res_tup[1], res_tup[2],
#    res_tup[3], res_tup[11], res_tup[12], res_tup[13], res_tup[14])
#        for size_value, res_tup in classifier_results.iteritems()]
#res_tups.sort()

def _get_ambiguity_data(classifier_results):
    # Get the ambiguity data for the res_tups (yuck...)
    res_tups = [(size_value, res_tup[11], res_tup[12])
            for size_value, res_tup in classifier_results.iteritems()]
    res_tups.sort()

    sample_sizes = [t[0] for t in res_tups]
    ambiguity_means = [t[1] for t in res_tups]
    ambiguity_stds = [t[2] for t in res_tups]

    return sample_sizes, ambiguity_means, ambiguity_stds

def _ambiguity_plot_gen(results, classifiers, datasets):
    #plt.ylabel('Accuracy')
    #plt.xlabel('Training Examples')
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    # Plot for each dataset and classifier
    for dataset in datasets:
        for classifier in classifiers:
            classifier_results = results[dataset][classifier]

            sample_sizes, ambiguity_means, ambiguity_stds = (
                    _get_ambiguity_data(classifier_results))

            fig = Figure()
            plt.errorbar(sample_sizes, ambiguity_means,
                    yerr=ambiguity_stds,
                    figure=fig,
                    label=classifier,
                    # Here we use style by classifier, rather than dataset
                    linestyle=LINE_STYLE_BY_CLASSIFIER[classifier],
                    color=LINE_COLOUR_BY_CLASSIFIER[classifier],
                    )
            yield 'amb_{}_{}'.format(classifier.lower(), dataset.lower()), fig
            plt.clf()

    # Plot by dataset and then all classifiers
    for dataset in datasets:
        fig = Figure()
        for classifier in classifiers:
            classifier_results = results[dataset][classifier]

            sample_sizes, ambiguity_means, ambiguity_stds = (
                    _get_ambiguity_data(classifier_results))

            plt.errorbar(sample_sizes, ambiguity_means,
                    yerr=ambiguity_stds,
                    figure=fig,
                    label=classifier,
                    # Here we use style by classifier, rather than dataset
                    linestyle=LINE_STYLE_BY_CLASSIFIER[classifier],
                    marker=LINE_MARKER_BY_CLASSIFIER[classifier],
                    color=LINE_COLOUR_BY_CLASSIFIER[classifier],
                    )

        yield 'amb_by_classifier', fig
        plt.clf()

    # XXX: Cut-off since some datasets are "too" big
    dataset_size_cut_off = True

    # Plot by classifier for all datasets
    for classifier in classifiers:
        fig = Figure()
        added = []
        for dataset in datasets:
            classifier_results = results[dataset][classifier]
            
            sample_sizes, ambiguity_means, ambiguity_stds = (
                    _get_ambiguity_data(classifier_results))

            line = plt.errorbar(sample_sizes, ambiguity_means,
                    # NOTE: Becomes pretty much unreadable
                    #yerr=ambiguity_stds,
                    figure=fig,
                    label=dataset,
                    # Here we use style by dataset, rather than classifier
                    linestyle=LINE_STYLE_BY_DATASET[dataset],
                    marker=LINE_MARKER_BY_DATASET[dataset],
                    color=LINE_COLOUR_BY_DATASET[dataset],
                    )
            added.append((dataset, line))

        # Legend handling (yuck...)
        #print fig.get_axes()
        #ax = fig.get_axes()[0]
        #handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(added))
        plt.legend(handles, labels)
        #ax.legend(handles, labels, loc=4)
        
        plt.xlim(xmax=10000)

        yield 'amb_by_dataset', fig
        plt.clf()

# XXX: God damn it, how do you do non-stateful matplotlib...!
def plot_learning_curve_results(classifiers, datasets, outdir,
        pickle_name='learning', verbose=False):
    from matplotlib import pyplot as plt

    if not classifiers and not datasets:
        print >> stderr, 'No classifiers or datasets specified, exiting'
        return

    # Load the results to be plotted
    if verbose:
        print >> stderr, 'Loading results...',
    with open(_get_learning_pickle_path(outdir, name=pickle_name), 'r'
            ) as results_file:
        results = pickle_load(results_file)
    if verbose:
        print >> stderr, 'Done!'

    image_formats = ('svg', )

    for fig_name, fig in _ambiguity_plot_gen(results, classifiers, datasets):
        for image_format in image_formats:
            plt.savefig(path_join(outdir, fig_name + '.' + image_format),
                    format=image_format, figure=fig) 

    # Over all classifiers for one dataset
    # Over all datasets for one classifier
    # Same as the above for ambiguity and recall

    #_plot_curve(outdir, results, pickle_name)
    #_plot_curve(outdir, results, pickle_name + '_ambiguity', new_metric=True)
    #_plot_curve(outdir, results, pickle_name + '_recall', new_metric=True, recall=True)
