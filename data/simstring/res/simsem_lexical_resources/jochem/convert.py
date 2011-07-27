#!/usr/bin/env python

'''
Extract CHEMICAL and CHEBI entries from the Jochem ErasmusMC ontology file.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-07-27
'''

from sys import stdin

def main(args):
    db_data = {}
    db_data['CHEMICAL'] = set()
    db_data['CHEBI'] = set()

    curr_name = None
    curr_dbs = set()
    curr_vars = set()
    for line in (l.rstrip('\n') for l in stdin):
        # If we enter a new entity, store and then reset everything
        if line == '--':
            for db in (_db for _db in curr_dbs if _db in db_data):
                if curr_name is not None:
                    db_data[db].add(curr_name)

                    for variant in curr_vars:
                        db_data[db].add(variant)
                    # TODO: More?

            curr_name = None
            curr_dbs = set()
            curr_vars = set()
        elif line.startswith('NA '):
            curr_name = line[3:]
        elif line.startswith('TM '):
            entry = line[3:].strip()
            if '@match' in entry:
                entry = entry[:entry.find('@match')].strip()
            curr_vars.add(entry)
        elif line.startswith('VO '):
            curr_dbs.add(line[3:])

    # Write to disk
    for db in db_data:
        with open('jochem_%s.tokens' % db.lower(), 'w') as tokens_file:
            tokens = [t for t in db_data[db]]
            tokens.sort()
            for token in tokens:
                tokens_file.write(token + '\n')

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
