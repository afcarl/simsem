#!/usr/bin/env python

'''
N-gram related, sorting, creation, etc.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-09-02
'''

from math import sqrt

def n_gram_ref_cos_cmp(a, b, ref_grams, n=3):
    a_grams = set(g for g in n_gram_gen(a, n=n))
    b_grams = set(g for g in n_gram_gen(b, n=n))
    return cmp(_ngram_cos_cmp(a_grams, ref_grams), _ngram_cos_cmp(b_grams, ref_grams))

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
def n_gram_gen(s, n=3):
    i = 0
    while i <= len(s) - n:
        yield s[i:i + n]
        i += 1

if __name__ == '__main__':
    from itertools import permutations

    s = 'python'
    print 'Tri-grams:'
    for g in n_gram_gen(s):
        print g

    print 'Full-string-gram:'
    for g in n_gram_gen(s, n=len(s)):
        print g

    print 'Too-long-gram:'
    for g in n_gram_gen(s, n=len(s) + 1):
        print g

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
