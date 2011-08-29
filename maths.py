'''
Mathematical related functions.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-08-29
'''

from math import sqrt

def mean(l):
    return sum(l) / float(len(l))

def median(l):
    return sorted(l)[len(l)/2]

def stddev(vals):
    avg = mean(vals)
    return sqrt(sum(((val - avg) ** 2 for val in vals)) / len(vals))

def truncated_mean(l, truncation=0.05):
    offset = int(len(l) * truncation)
    return sum(sorted(l)[offset:-offset]) / float(len(l) - 2 * offset)
