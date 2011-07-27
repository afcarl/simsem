#!/usr/bin/env python

'''
Generate a list of disease tokens from the AZDC corpus.

Example:

    ./azdctokens.py ${PATH_TO_AZDC_DATA} | sort | uniq > azdc.tokens

Author:     Pontus Stenetorp    pontus is s u-tokyo ac jp
Version:    2011-04-01
'''

from csv import DictReader
from os.path import dirname
from os.path import join as join_path

def main(args):
    azdc_path = args[1]

    with open(azdc_path, 'r') as azdc_file:
        for row in DictReader(azdc_file, delimiter='\t'):
            try:
                start = int(row['Start Point'])
                end = int(row['End Point'])

                # XXX: Start point should have had a -1 for accurate extraction!
                token = row['Sentence'][
                        int(row['Start Point']) - 1:
                        int(row['End Point'])].strip()
                if token:
                    print token
            except ValueError:
                # No annotation for this line
                pass

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
