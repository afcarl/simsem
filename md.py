#!/usr/bin/env python

'''
Some Markdown helpers to print readable and parseable data.

URL(s):

* Markdown: http://daringfireball.net/projects/markdown/
* GitHub-flavoured Markdown: http://github.github.com/github-flavored-markdown/
'''

from itertools import izip
from pprint import pprint

def dict_to_table(dic, total=True, perc=True):
    #pprint(dict(dic))
    x_lbls = sorted(x_l for x_l in dic)

    # This makes it possible to use sparse defaultdicts
    y_lbls = set()
    for lbls in (dic[x_l] for x_l in x_lbls):
        for lbl in lbls:
            y_lbls.add(lbl)
    y_lbls = sorted([l for l in y_lbls])

    hdr = ['Source / Target'] + x_lbls
    lns = [[y_lbl] + [dic[x_lbl][y_lbl] for x_lbl in x_lbls] for y_lbl in y_lbls]

    tot = (['**Total:**'] + [sum(lns[ln_i][c_i + 1]
        for ln_i in xrange(len(lns))) for c_i in xrange(len(x_lbls))])

    if perc:
        # Inject percentages
        for c_i, c_tot in izip(xrange(1, len(lns[0])),
                (t for i, t in enumerate(tot) if i > 0)):
            for ln_i in xrange(len(lns)):
                val = lns[ln_i][c_i]
                p = val / float(c_tot)
                # TODO: We would want the percentages right-adjusted
                lns[ln_i][c_i] = '{} ({:.2f}%)'.format(val, p)

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

if __name__ == '__main__':
    from collections import defaultdict
    from random import randint

    print dict_to_table(
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
    print

    ddic = defaultdict(lambda : defaultdict(int))

    lbls = [str(_s) for _s in xrange(4711, 4711 + 5)]
    for l in lbls:
        for _l in lbls:
            if randint(0, 1) == 1:
                ddic[l][_l] = randint(7, 17)
    print dict_to_table(ddic)

