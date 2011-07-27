#!/usr/bin/env python

'''
Convert UMLS into sub-files on a per-concept basis.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-07-27
'''

from sys import stdin

def _fname(string):
    return ('umls_' + string.lower()
            .replace(' ', '_')
            .replace(',', '_')
            .replace('.', '_')
            + '.list')

def main(args):
    handle_by_fname = {}

    for line in (l.rstrip('\n') for l in stdin):
        entry, _, _ , dicname, _ = line.split('|')

        fname = _fname(dicname)
        try:
            f_handle = handle_by_fname[fname]
        except KeyError:
            f_handle = open(fname, 'w')
            handle_by_fname[fname] = f_handle

        f_handle.write(entry + '\n')

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
