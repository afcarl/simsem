# By Sampo Pyysalo

import re

def findhead(np):
    # Simple algorithm for heuristically identifying the likely head of
    # the given noun phrase. Largely follows the method of Cohen et al.,
    # BioNLP'11. Returns strings before, head, after, where
    # before+head+after == np and head is the heuristically guessed head
    # word.
    initial_np = np

    # clean up and store and initial or terminal space
    start_space, end_space = "", ""
    m = re.match(r'^(\s+)(.*\s*)$', np)
    if m:
        start_space, np = m.groups()
    m = re.match(r'(.*?)(\s+)$', np)
    if m:
        np, end_space = m.groups()
    
    # start by splitting by first preposition occurring after
    # non-empty string, if any (not a "complete" list)
    m = re.match(r'^(.+?)( +(?:of|in|by|as|on|at|to|via|for|with|that|than|from|into|upon|after|while|during|within|through|between|whereas|whether) .*)$', np)
    if m:
        np, post_after = m.groups()
    else:
        np, post_after = np, ""

    # remove highly likely initial determiner followed by candidate
    # word, if any
    m = re.match(r'^((?:a|the)\s+)(\S.*)$', np)
    if m:
        pre_before, np = m.groups()
    else:
        pre_before, np = "", np

    # then, pick last "word" in the (likely) head NP, where "word" is
    # defined as space-separated non-space sequence containing at
    # least one alphabetic character, or the last non-space if not
    # found (or just the whole thing as a final fallback).
    m = re.match(r'^(.*\s|.*?)(\S*[a-zA-Z]\S*)(.*)$', np)
    if m:
        before, head, after = m.groups()
    else:
        m = re.match(r'^(.* )(\S+)(.*)$', np)
        if m:
            before, head, after = m.groups()
        else:
            before, head, after = "", np, ""

    # combine back possible omitted bits
    before = start_space + pre_before + before
    after  = after + post_after + end_space

    # sanity check
    assert before+head+after == initial_np, "INTERNAL ERROR: '%s' + '%s' + '%s' != '%s'" % (before, head, after, initial_np)

    return before, head, after
