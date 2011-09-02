#!/usr/bin/env python

'''
N-gram related, sorting, creation, etc.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-09-02
'''

from math import sqrt

### Constants
# Tilde is still ASCII and very uncommon (PubMed)
DEFAULT_START_GUARD = u'~'
DEFAULT_END_GUARD = u'~'
###

def n_gram_ref_cos_cmp(a, b, ref_grams, n=3, guards=False):
    a_grams = set(g for g in n_gram_gen(a, n=n, guards=guards))
    b_grams = set(g for g in n_gram_gen(b, n=n, guards=guards))
    return cmp(_ngram_cos_cmp(a_grams, ref_grams),
            _ngram_cos_cmp(b_grams, ref_grams))

def _ngram_cos_cmp(a, b):
    hits = 0
    misses = 0
    for g in a:
        if g in b:
            hits += 1
        else:
            misses += 1
    return hits / sqrt(len(a) * len(b))

# TODO: Start/End Guards
def n_gram_gen(s, n=3, guards=False, start_guard=DEFAULT_START_GUARD,
        end_guard=DEFAULT_END_GUARD):
    if guards:
        _s = start_guard + s + end_guard
    else:
        _s = s

    i = 0
    while i <= len(_s) - n:
        yield _s[i:i + n]
        i += 1

if __name__ == '__main__':
    from itertools import permutations

    s = 'python'
    print 'Tri-grams:'
    for g in n_gram_gen(s):
        print g
    print

    print 'Full-string-gram:'
    for g in n_gram_gen(s, n=len(s)):
        print g
    print

    print 'Too-long-gram:'
    for g in n_gram_gen(s, n=len(s) + 1):
        print g
    print
        
    print 'Tri-grams (guarded):'
    for g in n_gram_gen(s, guards=True):
        print g
    print

    print 'Sorting:'
    ref_grams = set(g for g in n_gram_gen(s))
    s_1 = 'pythonista'
    s_2 = 'pyts'
    s_3 = 'bullocks'
    s_4 = 'pythons'
    l = [s_2, s_1, s_3, s_4]
    print l
    for a, b in permutations(l, 2):
        v = n_gram_ref_cos_cmp(a, b, ref_grams)
        if v > 0:
            c = '>'
        elif v < 0:
            c = '<'
        else:
            c = '='

        print 'cmp: {} {} {}'.format(a, c, b)
    print sorted(l)
    print sorted(l, cmp=lambda a, b: -n_gram_ref_cos_cmp(a, b, ref_grams))
    print
    
    print 'Sorting (guarded):'
    unguarded_ref_grams = ref_grams
    ref_grams = set(g for g in n_gram_gen(s, guards=True))
    s_1 = 'python'
    s_2 = 'nythop'
    s_3 = 'bullocks'
    l = [s_2, s_1, s_3]
    print l
    for a, b in permutations(l, 2):
        v = n_gram_ref_cos_cmp(a, b, ref_grams, guards=True)
        if v > 0:
            c = '>'
        elif v < 0:
            c = '<'
        else:
            c = '='

        print 'cmp: {} {} {}'.format(a, c, b)
    print sorted(l, cmp=lambda a, b: -n_gram_ref_cos_cmp(a, b,
        unguarded_ref_grams, guards=False))
    print sorted(l, cmp=lambda a, b: -n_gram_ref_cos_cmp(a, b,
        ref_grams, guards=True))
