#!/usr/bin/env python

'''
Features and methods for extracting them.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2010-03-07
'''

from sys import path as sys_path
from operator import itemgetter
from itertools import imap, groupby, combinations
from os.path import join as join_path
from os.path import dirname, basename
from os import listdir

try:
    from cPickle import load as pickle_load
except ImportError:
    from pickle import load as pickle_load

from toolsconf import LIB_PATH, SIMSTRING_LIB_PATH

sys_path.append(LIB_PATH)
from porter import PorterStemmer

sys_path.append(SIMSTRING_LIB_PATH)
# TODO: Import simstring here instead

### Constants
PICKLE_DIR = join_path(dirname(__file__), 'data/pickle')
SIMSTRING_DBS_DIR = join_path(dirname(__file__), 'data/simstring')
SIMSTRING_DBS = [join_path(SIMSTRING_DBS_DIR, p)
        for p in listdir(SIMSTRING_DBS_DIR) if p.endswith('.db')]
###

#TODO: Always prefix the ret string with id + '-'!
#TODO: We need feature groups too, since some depend on each other
class AbstractFeature(object):
    def get_id(self):
        raise NotImplementedError

    def featurise(self, document, sentence, annotation):
        raise NotImplementedError


# TODO: Can be improved!
class EnsembleFeature(AbstractFeature):
    def __init__(self):
        raise NotImplementedError
    
    def featurise(self, document, sentence, annotation):
        # TODO: Catch him here if the AttributeError is raised
        for ensemble_feature in self.ensemble:
            for feature in ensemble_feature.featurise(document,
                    sentence, annotation):
                yield feature


#TODO: These guys can inherit shitloads of stuff! Replacing _ with space etc.
class StringPorterStemFeature(object):
    def __init__(self):
        self._stemmer = PorterStemmer()
        
    def _stem(self, token):
        return self._stemmer.stem(token, 0, len(token) - 1)
        return self._stemmer.stem(token, 0, len(token) - 1)

    def get_id(self):
        return 'STRING-STEM'

    def featurise(self, document, sentence, annotation):
        yield (self._stem(sentence.annotation_text(annotation)), 1)


class StringFeature(object):
    def get_id(self):
        return 'STRING'

    def featurise(self, document, sentence, annotation):
        yield (sentence.annotation_text(annotation), 1)


class LowerCaseStringFeature(object):
    def get_id(self):
        return 'STRING-LOWERCASE'

    def featurise(self, document, sentence, annotation):
        yield (sentence.annotation_text(annotation).lower(), 1)


from itertools import izip


class WindowStringFeature(object):
    def get_id(self):
        return 'WINDOW-STRING'

    def featurise(self, document, sentence, annotation):
        NORMALISE = True

        before_ann = sentence.text[:annotation.start].split()
        before_ann.reverse()
        after_ann = sentence.text[annotation.end:].split()

        to_yield = []
        for i, tok in izip(xrange(1, 4), before_ann):
            to_yield.append((u'-BEFORE-{}-{}'.format(i, tok), 1))
        for i, tok in izip(xrange(1, 4), after_ann):
            to_yield.append((u'-AFTER-{}-{}'.format(i, tok), 1))
        for f_tup in to_yield:
            if NORMALISE:
                yield (f_tup[0], f_tup[1] / float(len(to_yield)))
            else:
                yield f_tup



class WindowLowerCaseStringFeature(object):
    def get_id(self):
        return 'WINDOW-LOWERCASE-STRING'

    def featurise(self, document, sentence, annotation):
        NORMALISE = True

        before_ann = sentence.text[:annotation.start].split()
        before_ann.reverse()
        after_ann = sentence.text[annotation.end:].split()

        to_yield = []
        for i, tok in izip(xrange(1, 4), before_ann):
            to_yield.append((u'-BEFORE-{}-{}'.format(i, tok.lower()), 1))
        for i, tok in izip(xrange(1, 4), after_ann):
            to_yield.append((u'-AFTER-{}-{}'.format(i, tok.lower()), 1))
        for f_tup in to_yield:
            if NORMALISE:
                yield (f_tup[0], f_tup[1] / float(len(to_yield)))
            else:
                yield f_tup


#TODO: We can experiment with different lengths for these
class PrefixStringFeature(object):
    def get_id(self):
        return 'STRING-PREFIX'

    def featurise(self, document, sentence, annotation):
        ann_text = sentence.annotation_text(annotation)
        for i in xrange(3, 6):
            yield (ann_text[:i], 1)


class WindowPrefixStringFeature(object):
    def get_id(self):
        return 'WINDOW-PREFIX-STRING'

    def featurise(self, document, sentence, annotation):
        NORMALISE = True

        before_ann = sentence.text[:annotation.start].split()
        before_ann.reverse()
        after_ann = sentence.text[annotation.end:].split()
        
        to_yield = []
        for i, tok in izip(xrange(1, 4), before_ann):
            for j in xrange(3, 6):
                to_yield.append((u'-BEFORE-{}-{}'.format(i, tok[:j]), 1))
        for i, tok in izip(xrange(1, 4), after_ann):
            for j in xrange(3, 6):
                to_yield.append((u'-AFTER-{}-{}'.format(i, tok[:j]), 1))
        for f_tup in to_yield:
            if NORMALISE:
                yield (f_tup[0], f_tup[1] / float(len(to_yield)))
            else:
                yield f_tup


class SuffixStringFeature(object):
    def get_id(self):
        return 'STRING-SUFFIX'

    def featurise(self, document, sentence, annotation):
        ann_text = sentence.annotation_text(annotation)
        for i in xrange(3, 6):
            yield (ann_text[-i:], 1)


class WindowSuffixStringFeature(object):
    def get_id(self):
        return 'WINDOW-SUFFIX-STRING'

    def featurise(self, document, sentence, annotation):
        NORMALISE = True

        before_ann = sentence.text[:annotation.start].split()
        before_ann.reverse()
        after_ann = sentence.text[annotation.end:].split()

        to_yield = []
        for i, tok in izip(xrange(1, 4), before_ann):
            for j in xrange(3, 6):
                to_yield.append((u'-BEFORE-{}-{}'.format(i, tok[-j:]), 1))
        for i, tok in izip(xrange(1, 4), after_ann):
            for j in xrange(3, 6):
                to_yield.append((u'-AFTER-{}-{}'.format(i, tok[-j:]), 1))
        for f_tup in to_yield:
            if NORMALISE:
                yield (f_tup[0], f_tup[1] / float(len(to_yield)))
            else:
                yield f_tup


# TODO: We use a very primitive tokenisation... More finegrained?
class SentenceStringFeature(object):
    def get_id(self):
        return 'STRING-SENTENCE'

    def featurise(self, document, sentence, annotation):
        for token in sentence.text.split():
            yield (token, 1)


class SentenceLowerCaseStringFeature(object):
    def get_id(self):
        return 'STRING-SENTENCE-LOWERCASE'

    def featurise(self, document, sentence, annotation):
        for token in sentence.text.split():
            yield (token.lower(), 1)


class SentencePorterStemStringFeature(object):
    def __init__(self):
        self._stemmer = PorterStemmer()

    def _stem(self, token):
        return self._stemmer.stem(token, 0, len(token) - 1)

    def get_id(self):
        return 'STRING-SENTENCE-STEM'

    def featurise(self, document, sentence, annotation):
        for token in sentence.text.split():
            yield (self._stem(token), 1)


class WindowPorterStemStringFeature(object):
    def __init__(self):
        self._stemmer = PorterStemmer()

    def get_id(self):
        return 'WINDOW-STEM-STRING'
    
    def _stem(self, token):
        return self._stemmer.stem(token, 0, len(token) - 1)

    def featurise(self, document, sentence, annotation):
        NORMALISE = True

        before_ann = sentence.text[:annotation.start].split()
        before_ann.reverse()
        after_ann = sentence.text[annotation.end:].split()

        to_yield = []
        for i, tok in izip(xrange(1, 4), before_ann):
            to_yield.append((u'-BEFORE-{}-{}'.format(i, self._stem(tok)), 1))
        for i, tok in izip(xrange(1, 4), after_ann):
            to_yield.append((u'-AFTER-{}-{}'.format(i, self._stem(tok)), 1))
        for f_tup in to_yield:
            if NORMALISE:
                yield (f_tup[0], f_tup[1] / float(len(to_yield)))
            else:
                yield f_tup



### REG FROM HERE!
#TODO: Freq.
#TODO: Pos in sentence
#TODO: Pos of sentence in sentences
#TODO: Number of parenthesis (contains too)
#TODO: Number of brackets (contains too)
#TODO: Inside quotes, parens, brackets
#TODO: Contains upper, lower, digit, hyhen, period, single quote, ampersand
#TODO: Is camelcase
#TODO: Grams with start and end anchors!
#TODO: Doc frequency
#TODO: Training data frequency
#TODO: Doc-Permanence
#TODO: Lemma

def _collins_pattern(char):
    if char.isalpha():
        if char.isupper():
            return 'A'
        else:
            return 'a'
    elif char.isdigit():
        return '0'
    else:
        return '-'


class CollinsPatternStringFeature(object):
    def get_id(self):
        return 'STRING-COLLINS-PATTERN'

    def featurise(self, document, sentence, annotation):
        yield (''.join((_collins_pattern(c)
                for c in sentence.annotation_text(annotation))), 1)


class SentenceCollinsPatternStringFeature(object):
    def __init__(self):
        self._stemmer = PorterStemmer()

    def get_id(self):
        return 'STRING-COLLINS-STEM'

    def featurise(self, document, sentence, annotation):
        for token in sentence.text.split():
            yield (''.join((_collins_pattern(c) for c in token)), 1)
           
from itertools import groupby

class CollapsedCollinsPatternStringFeature(object):
    def get_id(self):
        return 'STRING-COLLAPSED-COLLINS-PATTERN'

    def featurise(self, document, sentence, annotation):
        yield (''.join((k for k, _ in groupby(''.join((_collins_pattern(c)
                for c in sentence.annotation_text(annotation)))))), 1)

class SentenceCollapsedCollinsPatternStringFeature(object):
    def __init__(self):
        self._stemmer = PorterStemmer()

    def get_id(self):
        return 'STRING-COLLINS-STEM'

    def featurise(self, document, sentence, annotation):
        for token in sentence.text.split():
            yield (''.join((k for k, _ in groupby(''.join((_collins_pattern(c) for c in token))))), 1)


class WindowCollinsPatternStringFeature(object):
    def get_id(self):
        return 'WINDOW-COLLIN-STRING'
    
    def featurise(self, document, sentence, annotation):
        NORMALISE = True

        before_ann = sentence.text[:annotation.start].split()
        before_ann.reverse()
        after_ann = sentence.text[annotation.end:].split()

        to_yield = []
        for i, tok in izip(xrange(1, 4), before_ann):
            to_yield.append(('-BEFORE-{}-{}'.format(i,
                ''.join((_collins_pattern(c) for c in tok))), 1))
        for i, tok in izip(xrange(1, 4), after_ann):
            to_yield.append(('-AFTER-{}-{}'.format(i,
                ''.join((_collins_pattern(c) for c in tok))), 1))
        for f_tup in to_yield:
            if NORMALISE:
                yield (f_tup[0], f_tup[1] / float(len(to_yield)))
            else:
                yield f_tup

            
class WindowCollapsedCollinsPatternStringFeature(object):
    def get_id(self):
        return 'WINDOW-COLLAPSED-COLLIN-STRING'
    
    def featurise(self, document, sentence, annotation):
        NORMALISE = True

        before_ann = sentence.text[:annotation.start].split()
        before_ann.reverse()
        after_ann = sentence.text[annotation.end:].split()

        to_yield = []
        for i, tok in izip(xrange(1, 4), before_ann):
            to_yield.append(('-BEFORE-{}-{}'.format(i,
                ''.join(k for k, _ in groupby(
                ''.join((_collins_pattern(c) for c in tok))))), 1))
        for i, tok in izip(xrange(1, 4), after_ann):
            to_yield.append(('-AFTER-{}-{}'.format(i,
                ''.join(k for k, _ in groupby(
                ''.join((_collins_pattern(c) for c in tok))))), 1))
        for f_tup in to_yield:
            if NORMALISE:
                yield (f_tup[0], f_tup[1] / float(len(to_yield)))
            else:
                yield f_tup

                
class FormattingStringFeature(object):
    def get_id(self):
        return 'FORMATTING-STRING'

    def _featurise(self, token):
        # "Type"
        if all(c.isupper() for c in token):
            yield ('TYPE-ALLUPPER', 1)
        elif all(c.isdigit() for c in token):
            yield ('TYPE-ALLDIGIT', 1)
        elif all(not c.isalpha() and not c.isdigit() for c in token):
            yield ('TYPE-ALLSYMBOL', 1)
        elif all(c.isupper() or c.isdigit() for c in token):
            yield ('TYPE-ALLUPPERDIGIT', 1)
        elif all(c.isupper() or (not c.isalpha() and not c.isdigit())
                for c in token):
            yield ('TYPE-ALLUPPERSYMBOL', 1)
        elif all(c.isupper() or c.isdigit()
                or (not c.isalpha() and not c.isdigit()) for c in token):
            yield ('TYPE-ALLUPPERDIGITSYMBOL', 1)
        elif token[0].isupper():
            yield ('TYPE-INITUPPER', 1)
        elif all(c.isalpha() for c in token):
            yield ('TYPE-ALLLETTER', 1)
        elif all(c.isalpha() or c.isdigit() for c in token):
            yield ('TYPE-ALLLETTERDIGIT', 1)

        if len(token) == 2 and all(c.isdigit() for c in token):
            yield ('TWODIGIT', 1)
        elif len(token) == 4 and all(c.isdigit() for c in token):
            yield ('FOURDIGIT', 1)

        if all(c.isdigit() or c.isalpha() for c in token):
            yield ('DIGITALPHA', 1)

        if all(c.isdigit() or c == '-' for c in token):
            yield ('DIGITHYPHEN', 1)

        if all(c.isdigit() or c == '/' for c in token):
            yield ('DIGITSLASH', 1)

        if all(c.isdigit() or c == ',' for c in token):
            yield ('DIGITCOLON', 1)

        if all(c.isdigit() or c == '.' for c in token):
            yield ('DIGITDOT', 1)

        if all(c.isupper() or c == '.' for c in token):
            yield ('UPPERDOT', 1)

        if DATE_REGEX.match(token) is not None:
            yield ('DATE', 1)

        if token[0].isupper():
            yield ('INITUPPER', 1)

        if all(c.isupper() for c in token):
            yield ('ALLUPPER', 1)

        if all(c.islower() for c in token):
            yield ('ALLLOWER', 1)

        if all(c.isdigit() for c in token):
            yield ('ALLDIGITS', 1)

        if all(not c.isdigit() and not c.isalpha() for c in token):
            yield ('ALLNONALPHANUM', 1)

        if any(c.isupper() for c in token):
            yield ('CONTAINUPPER', 1)

        if any(c.islower() for c in token):
            yield ('CONTAINLOWER', 1)

        if any(c.isalpha() for c in token):
            yield ('CONTAINALPHA', 1)

        if any(c.isdigit() for c in token):
            yield ('CONTAINDIGITS', 1)

        if any(not c.isalpha() or not c.isdigit() for c in token):
            yield ('CONTAINNONALPHANUM', 1)

    def featurise(self, document, sentence, annotation):
        token = sentence.annotation_text(annotation)

        if len(token) < 1:
            print sentence.text
            print annotation
            print token
            assert False

        for f_tup in self._featurise(token):
            yield f_tup


class WindowFormattingStringFeature(object):
    def __init__(self):
        self.feature = FormattingStringFeature()

    def get_id(self):
        return 'WINDOW-FORMATTING-STRING'
    
    def featurise(self, document, sentence, annotation):
        before_ann = sentence.text[:annotation.start].split()
        before_ann.reverse()
        after_ann = sentence.text[annotation.end:].split()

        for i, tok in izip(xrange(1, 4), before_ann):
            for f_tup in self.feature._featurise(tok):
                yield ('-BEFORE-{}-{}'.format(i, f_tup[0]), f_tup[1])
        for i, tok in izip(xrange(1, 4), after_ann):
            for f_tup in self.feature._featurise(tok):
                yield ('-AFTER-{}-{}'.format(i, f_tup[0]), f_tup[1])

from re import compile as _compile
DATE_REGEX = _compile(r'^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$')

#TODO: Window BoW!

### Features to capture NP internal performance

class SpanBoWFeature(object):
    def get_id(self):
        return 'SPAN-BOW'

    def featurise(self, document, sentence, annotation):
        span_text = sentence.annotation_text(annotation)

        for tok in span_text.split():
            yield (tok, 1)

from lib.findhead import findhead as find_np_head

class SpanHeadFeature(object):
    def get_id(self):
        return 'SPAN-HEAD'

    def featurise(self, document, sentence, annotation):
        span_text = sentence.annotation_text(annotation)
        yield (find_np_head(span_text)[1], 1)

class SpanHeadWindowFeature(object):
    def get_id(self):
        return 'SPAN-HEAD-WINDOW'

    def featurise(self, document, sentence, annotation):
        span_text = sentence.annotation_text(annotation)
        before_head, _, after_head = find_np_head(span_text)
        for tok_i, tok in enumerate(reversed(before_head.split(' ')), start=1):
            if tok_i > 3:
                break
            yield ('-BEFORE-{}-{}'.format(tok_i, tok), 1)
        for tok_i, tok in enumerate(after_head.split(' '), start=1):
            if tok_i > 3:
                break
            yield ('-AFTER-{}-{}'.format(tok_i, tok), 1)

###

### Trailing constants
# TODO: We can GENERATE a lot of the shit above here
SIMPLE_SPAN_INTERNAL_CLASSES = set((
        StringPorterStemFeature,
        StringFeature,
        LowerCaseStringFeature,
        PrefixStringFeature,
        SuffixStringFeature,
        ))
###

if __name__ == '__main__':
    raise NotImplementedError
    # We will do a test with a random document from the ID train set
    from reader.bionlp import get_id_set

    train, _, _ = get_id_set()
    
    # TODO: Make in random...
    document = next(train)

    features = [f() for f in FEATURE_CLASSES]

    for sentence in document:
        for annotation in sentence:
            for feature in features:
                for tup in feature.featurise(document, sentence, annotation):
                    print feature.get_id() + '-' + tup[0] + ':' + str(tup[1])
