#!/usr/bin/env python

'''
Useful helpers and functions that could not be put elsewhere.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-02-16
'''

def readable_file(string):
    '''
    argparser helper to require files given to exist and be readable

    Argument(s):
    string - string to be checked and returned if it is a file and is readable

    Raises:
    argparse.ArgumentTypeError - if the given string is not readable or a file
    '''
    from os.path import exists, isfile
    from os import access, R_OK
    from argparse import ArgumentTypeError

    if not isfile(string):
        if not exists(string):
            raise ArgumentTypeError(
                    '{}: no such file or directory'.format(string))
        else:
            raise ArgumentTypeError('{}: is not a file'.format(string))
    elif not access(string, R_OK):
        raise ArgumentTypeError('{}: is not a file'.format(string))
    else:
        return string

def writeable_dir(string):
    #XXX: TODO: Dir!
    from os.path import exists, isdir
    from os import access, W_OK
    from argparse import ArgumentTypeError
    
    if not isdir(string):
        if not exists(string):
            raise ArgumentTypeError(
                    '{}: no such file or directory'.format(string))
        else:
            raise ArgumentTypeError('{}: is not a directory'.format(string))
    elif not access(string, W_OK):
        raise ArgumentTypeError(
                '{}: is not a writeable directory'.format(string))
    else:
        return string
