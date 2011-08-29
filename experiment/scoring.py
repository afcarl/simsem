#!/usr/bin/env python

'''
Score classifiers according to various metrics.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-30
'''

from collections import defaultdict
from itertools import chain, izip

from maths import mean, median, truncated_mean

def score_classifier(classifier, test_set):
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
    micro_score = mean(micro_scores)

    return (macro_score, micro_score, tp_sum, fn_sum, results_by_class)

def score_classifier_by_tup(classifier, test_tups):
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
    micro_score = mean(micro_scores)

    return (macro_score, micro_score, tp_sum, fn_sum, results_by_class)
        
def score_classifier_by_tup_ranked(classifier, test_tups,
        conf_threshold=0.995, unseen=False):
    results_by_class = {}
    ambd_by_class = defaultdict(list)
    not_in_range_by_class = defaultdict(int)
    for test_lbl, test_vec in izip(*test_tups):
        if not isinstance(test_lbl, str):
            test_lbl_type = classifier.name_by_lbl_id[test_lbl]
        else:
            test_lbl_type = test_lbl

        # TODO: Cast annotation into span! It needs to be censored
        # XXX: Ranked will fail for some classifiers since not all are prob.
        predicted = classifier._classify(test_vec, ranked=True)

        # Find where the correct answer was ranked
        for i, pred in enumerate((p for p, _ in predicted), start=1):
            if pred == test_lbl_type:
                rank = i
                break
        else:
            # If there are unseen categories our assection can not hold
            if not unseen:
                assert False, "'{}' not in {}".format(test_lbl_type,
                        predicted) # Should not happen
            else:
                rank = i

        conf_sum = 0.0
        for i, prob in enumerate((p for _, p in predicted), start=1):
            conf_sum += prob
            if conf_sum >= conf_threshold:
                conf_threshold_cutoff = i
                break
        else:
            conf_threshold_cutoff = i
        ambd_by_class[test_lbl_type].append(conf_threshold_cutoff)

        # Determine if the threshold cut away the correct answer
        if rank > conf_threshold_cutoff:
            not_in_range_by_class[test_lbl_type] += 1

        try:
            results_by_class[test_lbl_type].append(rank)
        except KeyError:
            results_by_class[test_lbl_type] = [rank, ]

    ranks = [e for e in chain(*results_by_class.itervalues())]
    lost_by_threshold = sum(not_in_range_by_class.itervalues()) / float(len(ranks))
    avg_ambiguity_size = mean([e for e in chain(*ambd_by_class.itervalues())])
    mean_rank = mean(ranks)
    median_rank = median(ranks)
    # 5% on each side
    truncated_mean_rank = truncated_mean(ranks)

    return (mean_rank, median_rank, truncated_mean_rank, avg_ambiguity_size, lost_by_threshold)
