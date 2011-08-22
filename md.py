#!/usr/bin/env python

'''
Some Markdown helpers to print readable and parseable data.

URL(s):

* Markdown: http://daringfireball.net/projects/markdown/
* GitHub-flavoured Markdown: http://github.github.com/github-flavored-markdown/
'''

from itertools import izip
from pprint import pprint

def _get_lbls(dic):
    x_lbls = [str(x_l) for x_l in dic]

    # This makes it possible to use sparse defaultdicts
    y_lbls = set()
    for lbls in (dic[x_l] for x_l in x_lbls):
        for lbl in lbls:
            y_lbls.add(lbl)
    y_lbls = [l for l in y_lbls]

    return x_lbls, y_lbls

def dict_to_table(dic, total=True, perc=True, sorted=True, hdr_lbl=''):
    x_lbls, y_lbls = _get_lbls(dic)

    if sorted:
        try:
            for lbl in x_lbls:
                int(lbl)
            x_lbls.sort(key=int)
        except ValueError:
            x_lbls.sort()

        try:
            for lbl in y_lbls:
                int(lbl)
            y_lbls.sort(key=int)
        except ValueError:
            y_lbls.sort()

    hdr = [hdr_lbl] + x_lbls
    lns = [[y_lbl] + [dic[x_lbl][y_lbl]
        for x_lbl in x_lbls] for y_lbl in y_lbls]

    tot = ['**Total:**', ]
    for c_i in xrange(1, len(x_lbls) + 1):
        seen_num = False
        c_sum = 0
        for ln_i in xrange(len(lns)):
            try:
                val = lns[ln_i][c_i]
                c_sum += val
                seen_num = True
            except TypeError:
                pass

        if not seen_num:
            tot.append('-')
        else:
            tot.append(c_sum)

    if perc:
        # Inject percentages
        for c_i, c_tot in izip(xrange(1, len(lns[0])),
                (t for i, t in enumerate(tot) if i > 0)):
            if c_tot == '-':
                continue

            for ln_i in xrange(len(lns)):
                val = lns[ln_i][c_i]
                try:
                    p = val / float(c_tot)
                    # TODO: We would want the percentages right-adjusted
                    lns[ln_i][c_i] = '{} ({:.2f}%)'.format(val, p)
                except TypeError:
                    # It was not a number for that value, ignore it
                    pass

    if total:
        lns.append([str(t) for t in tot])

    # Expand each column to fit in size
    stx = []
    for c_i in xrange(len(hdr)):
        max_c_len = len(hdr[c_i])
        for ln_i in xrange(len(lns)):
            # Convert to a string in case it isn't, the math is done at this point
            #print c_i, ln_i
            lns[ln_i][c_i] = str(lns[ln_i][c_i])
            val = lns[ln_i][c_i]
            max_c_len = max(len(val), max_c_len)

        # A small read-ability offset
        max_c_len += 2

        # Adjust the lengths
        if c_i > 0:
            hdr[c_i] = ' ' + hdr[c_i] + ' ' * (max_c_len - len(hdr[c_i]) - 1)
        else:
            hdr[c_i] = hdr[c_i] + ' ' * (max_c_len - len(hdr[c_i]) - 1)

        for ln_i in xrange(len(lns)):
            if c_i > 0:
                lns[ln_i][c_i] = ' ' + lns[ln_i][c_i] + ' ' * (max_c_len - len(lns[ln_i][c_i]) - 1)
            else:
                lns[ln_i][c_i] = lns[ln_i][c_i] + ' ' * (max_c_len - len(lns[ln_i][c_i]) - 1)


        if c_i > 0:
            stx.append(':--' + ('-' * (max_c_len - 3)))
        else:
            stx.append(':--' + ('-' * (max_c_len - 4)))

    return '\n'.join((
        '|'.join(hdr) + '|',
        '|'.join(stx) + '|',
        '\n'.join(
            '|'.join(ln) + '|' for ln in lns
            )
        ))

def trans_dict(dic):
    new_dic = {}

    x_lbls, y_lbls = _get_lbls(dic)

    for y_lbl in y_lbls:
        new_dic[y_lbl] = {}
        for x_lbl in x_lbls:
            new_dic[y_lbl][x_lbl] = dic[x_lbl][y_lbl]

    return new_dic

import re

def table_to_dict(table, strip_perc=True):
    dic = {}
    lns = table.split('\n')
    
    # Drop the second line since it is only syntax
    del lns[1]

    # Grab the labels and the data
    x_lbls = [lbl.strip() for lbl in lns[0].split('|')[1:-1]]
    y_lbls = [ln[:ln.find('|')].strip() for ln in lns[1:]]
    data = [[lbl.strip() for lbl in ln.split('|')[1:-1]] for ln in lns[1:]]

    # Remove the total if there is any
    if y_lbls[-1] == '**Total:**':
        y_lbls = y_lbls[:-1]
        data = data[:-1]

    if strip_perc:
        for ln_i in xrange(len(data)):
            for c_i in xrange(len(data[ln_i])):
                data[ln_i][c_i] = re.sub(
                        r'(.*?)\([0-9]+\.[0-9]+%\)?',
                        r'\1', data[ln_i][c_i]).strip()

    for c_i, x_lbl in enumerate(x_lbls):
        dic[x_lbl] = {}
        for y_lbl, ln in izip(y_lbls, data):
            dic[x_lbl][y_lbl] = ln[c_i]

    return dic

if __name__ == '__main__':
    from collections import defaultdict
    from random import randint

    tbl = dict_to_table(
            {
                'foo': {
                    'foo':   1,
                    'bar':   0,
                    'baren': 0,
                    },
                'bar': {
                    'foo':   1,
                    'bar':   1,
                    'baren': 0,
                    },
                'baz': {
                    'foo':   1,
                    'bar':   0,
                    'baren': 1,
                    },
                }
            )
    print tbl
    print

    ddic = defaultdict(lambda : defaultdict(int))

    lbls = [str(_s) for _s in xrange(4711, 4711 + 5)]
    for l in lbls:
        for _l in lbls:
            if randint(0, 1) == 1:
                ddic[l][_l] = randint(7, 17)
    tbl = dict_to_table(ddic)
    print tbl
    print
    
    tbl = dict_to_table(
            {
                'foo': {
                    'foo':   1,
                    'bar':   0,
                    'baren': '-',
                    },
                'bar': {
                    'foo':   1,
                    'bar':   1,
                    'baren': '-',
                    },
                'baz': {
                    'foo':   '-',
                    'bar':   '-',
                    'baren': '-',
                    },
                }
            )
    print tbl
    print

    dic = table_to_dict(tbl)
    pprint(dic)
    print

    tbl = dict_to_table(dic, total=False, perc=False)
    print tbl

    for x_lbl in dic:
        for y_lbl in dic[x_lbl]:
            val = dic[x_lbl][y_lbl]
            if val != '-':
                dic[x_lbl][y_lbl] = int(val)

    tbl = dict_to_table(dic)
    print tbl
    print

    print trans_dict(dic)
