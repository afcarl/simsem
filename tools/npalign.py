#!/usr/bin/env python

'''
Align text-bound stand-off annotations to noun phrases.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-11-02
'''

# XXX: This script is a mess, should be refactored:
#   * Handle ALL annotations at once, currently we don't do interactions
#       between annotations from a1 and a2
# XXX: We are currently ignoring coordination!

# TODO: Minimal NP;s, maximal NP;s heuristics
# TODO: Expand spans only
# TODO: Shrink spans only
# Default is to fit onto the NP
# TODO: fname is actually fpath

from argparse import ArgumentParser, FileType
from itertools import chain
from re import compile as re_compile
from string import whitespace
from sys import maxint
from sys import stderr

### Constants
ARGPARSER = ArgumentParser()
ARGPARSER.add_argument('text_file', type=FileType('r'))
ARGPARSER.add_argument('ptb_file', type=FileType('r'))
ARGPARSER.add_argument('stand_off_file', nargs='+', type=FileType('r'))
ARGPARSER.add_argument('-a', '--non-alpha-heuristic', action='store_true', help="if an np head is covered by annotations of the same type apart fron it's non-alpha characters create an annotation (example: \"p16(INK4a)\")")
ARGPARSER.add_argument('-n', '--no-warn', action='store_true')
ARGPARSER.add_argument('-g', '--generate', action='store_true',
        help=('generate additional annotations by eliminating determiners '
            '(DT), pronouns (PRP and PRP$) and cardinal numbers (CD) from '
            'the beginning of the annotation'))
ARGPARSER.add_argument('-m', '--merge', action='store_true')
ARGPARSER.add_argument('-d', '--debug', action='store_true')
ARGPARSER.add_argument('-r', '--dry-run', action='store_true')
ARGPARSER.add_argument('-v', '--verbose', action='store_true')
PTB_TAGS_REGEX = re_compile(r'\((?P<tag>[^ ]+)')
PTB_TOKENS_REGEX = re_compile(r'(?P<token>[^ )]+?)\)')
WHITESPACE_CHARS = set(whitespace)
###

def _ptb_token_gen(ptb):
    for match in PTB_TOKENS_REGEX.finditer(ptb):
        yield match.groupdict()['token']

def _ptb_tag_gen(ptb):
    for match in PTB_TAGS_REGEX.finditer(ptb):
        yield match.groupdict()['tag']

class Span(object):
    def __init__(self, start, end, type_=None, text=None, id_=None):
        self.start = start
        self.end = end
        self.type_ = type_
        self.text = text
        self.id_ = id_

    def __contains__(self, other):
        if isinstance(other, Span):
            return other.start >= self.start and other.end <= self.end
        else:
            return other >= self.start and other < self.end

    def __repr__(self):
        return str(self)

    def __hash__(self):
        # Note: We are leaving out id and text here
        return hash(hash(self.start) + hash(self.end) + hash(self.type_))

    def __cmp__(self, other):
        return other.start - self.start
    
    def __str__(self):
        return 'Span({}{}, {}{}{})'.format(
                self.type_ + ', ' if self.type_ is not None else '',
                self.start, self.end,
                ', ' + self.text if self.text is not None else '',
                ', ' + self.id_ if self.id_ is not None else '')

# NOTE: O(n), could be optimised if necessary
class Spans(object):
    def __init__(self, it=None):
        self.spans = []
        if it is not None:
            for span in it:
                self.spans.append(span)

    def add(self, span):
        self.spans.append(span)

    # Find the first overlapping span
    # TODO: Find all spans
    def find_all(self, other):
        for span in self:
            if other in span:
                yield span

    def __iter__(self):
        return iter(self.spans)

    def __contains__(self, other):
        for span in self:
            if other in span:
                return True
        else:
            return False

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Spans([' + ', '.join(str(s) for s in self.spans) + '])'

# TODO: function extracted from main

# Does not include quoting
PTB_ESCAPES = {
        '(': '-LRB-',
        ')': '-RRB-',
        '[': '-LSB-',
        ']': '-RSB-',
        '{': '-LCB-',
        '}': '-RCB-',
        }

PTB_SEXP_QUOTE_ESCAPES = {
        '(`` ")':   '(`` ``)',
        "('' \")": "('' '')",
        # XXX: Exception? Probably a failed parse...
        '(POS ")': "(POS '')",
        '(NN ")': "(NN '')",
        }

def _unescape(s):
    for _from, to in PTB_ESCAPES.iteritems():
        s = s.replace(to, _from)
    return s

def _sexp_quotes_unescape(s):
    #print 'YYY:', s
    for _from, to in PTB_SEXP_QUOTE_ESCAPES.iteritems():
        #print to, _from
        s = s.replace(to, _from)
    #print 'XXX:', s
    return s


def _token_i_mapping(tokens, text, text_start=0):
    
    #print >> stderr, 'text:', text
    #print >> stderr, 'tokens:', tokens

    token_i_to_offsets = {}
    text_pos = text_start

    for token_i, token in enumerate(tokens):
        
        #print >> stderr, 'token:', token

        token_start_txt_pos = text_pos
        token_pos = 0
        while token_pos < len(token):
            token_char = token[token_pos]
            text_char = text[text_pos]
            
            #print >> stderr, ('token_char: "{}" text_char: "{}"'
            #        ).format(token_char, text_char)

            if token_char == text_char:
                token_pos += 1
                text_pos += 1
            elif text_char in WHITESPACE_CHARS:
                text_pos += 1
                # If we are yet to begin matching the token it should not
                #   start with whitespace
                if token_pos == 0:
                    token_start_txt_pos += 1
            else:
                # Really nasty corner-case where the type of parentheses may
                #   have gone lost through PTB escaping etc.
                if (token_char == '"'
                        and text[text_pos:text_pos + 2] in ('``', "''", )):
                    # Skip ahead and assign this single token character as two
                    #   in the text
                    token_pos += 1
                    text_pos += 2
                else:
                    print >> stderr, 'ERROR: Failed to align token'
                    exit(-1) # XXX: Rude exit

        token_start = token_start_txt_pos
        token_end = text_pos
        token_i_to_offsets[token_i] = Span(token_start, token_end, text=token)

        # Token length sanity check, XXX: Can't be used with the corner-case above
        #assert len(token) == token_end - token_start, '"{}" {} {} {}'.format(
        #        token, token_start, token_end, token_end - token_start)
    
    return token_i_to_offsets, text_pos

def _paren_range(s):
    depth = 0
    for char_i, char in enumerate(s):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        if depth == 0:
            break
    else:
        print >> stderr, (
                'ERROR: Unmatched parentheses in S-expression')
        exit(-1) # XXX: Nasty exit
    return char_i + 1


def _create_token_spans(ptb_sexp, txt_data, token_i_to_offsets):
    tag_spans = []

    for match in PTB_TAGS_REGEX.finditer(ptb_sexp):
        tag = match.groupdict()['tag']

        # How many tokens proceed this tag?
        ptb_sexp_prior = ptb_sexp[:match.start()]
        tokens_prior = len(PTB_TOKENS_REGEX.findall(ptb_sexp_prior))

        # Find the S-expression span covered by the tag
        paren_end = _paren_range(ptb_sexp[match.start():])
        tag_start = match.start()
        tag_end = tag_start + (paren_end - 1)

        # Use the span to calculate the number of tokens covered
        ptb_sexp_segment = ptb_sexp[tag_start:tag_end + 1]
        tokens_contained = len(PTB_TOKENS_REGEX.findall(ptb_sexp_segment))

        tag_start_token_i = tokens_prior
        tag_end_token_i = tag_start_token_i + tokens_contained - 1
        
        tag_txt_start = token_i_to_offsets[tag_start_token_i].start
        tag_txt_end = token_i_to_offsets[tag_end_token_i].end
        tag_txt = txt_data[tag_txt_start:tag_txt_end]

        tag_span = Span(tag_txt_start, tag_txt_end, text=tag_txt, type_=tag)
        # Monkey patch the S-expression details in there
        tag_span.sexp = ptb_sexp_segment
        tag_span.sexp_start = tag_start
        tag_span.sexp_end = tag_end
        tag_spans.append(tag_span)

    return tag_spans

def _parse_tbs(stand_off_files):
    tb_spans_by_fname = {}

    for stand_off_file in stand_off_files:
        fname = stand_off_file.name
        tb_spans = []

        for line in (l.rstrip('\n') for l in stand_off_file):
            # Skip non-textbounds
            if not line.startswith('T'):
                continue
            
            # NOTE: Fails if no text is present
            id_, type_offsets, text = line.split('\t')
            type_, start, end = type_offsets.split(' ')
            start = int(start)
            end = int(end)

            tb_spans.append(Span(start, end, type_=type_, text=text, id_=id_))

        tb_spans_by_fname[fname] = tb_spans

    return tb_spans_by_fname

PRE_DROPS = set(('PRP', 'PRP$', 'DT', 'CD', ))

def main(args):
    argp = ARGPARSER.parse_args(args[1:])

    txt_data = argp.text_file.read()
    txt_pos = 0

    # Plural although a collection, oh dear... "stand_off_files"
    tb_spans_by_fname = _parse_tbs(argp.stand_off_file)
    
    from collections import defaultdict
    new_tb_spans_by_fname = defaultdict(list)
    
    expanded_ids_by_fname = defaultdict(set)

    tb_nums = (int(tb.id_[1:]) for tb in chain(*tb_spans_by_fname.itervalues()))
    next_tb_id = (i for i in xrange(max(chain((1, ), tb_nums)) + 1, maxint))

    # Restore the PTB quotes at the S-Expression stage or `` and '' fails
    for ptb_sexp_i, ptb_sexp in enumerate((l.rstrip('\n')
            for l in argp.ptb_file), start=1):
        if argp.debug:
            print >> stderr, ('Processing S-expression ({}): {}'
                    ).format(ptb_sexp_i, ptb_sexp)

        #print ptb_sexp
        quotes_unescaped_ptb_sexp = _sexp_quotes_unescape(ptb_sexp)
        #print >> stderr, quotes_unescaped_ptb_sexp
        #continue
        unescaped_tokens = [_unescape(t) for t in _ptb_token_gen(
            quotes_unescaped_ptb_sexp)]

        #print >> stderr, 'txt_data:', txt_data

        # Being unescaped the tokens should now align with the text
        token_i_to_offsets, txt_pos = _token_i_mapping(unescaped_tokens,
                txt_data, text_start=txt_pos)
    
        # Now calculate which tags covers which tokens
        tag_spans = _create_token_spans(quotes_unescaped_ptb_sexp, txt_data,
                token_i_to_offsets)
        sentence_tag = tag_spans[0]
        assert sentence_tag.type_ == 'S1'
    
        # XXX: How did we handle co-ordination?

        # Head-finding algorithm (similar to Bunescu & Mooney (2004)):
        # * Strip PP and VP from the right
        # * Once no further PP and/or VP can be stripped, the last noun;ish
        #       thing of the current NP is the head
        for fname in tb_spans_by_fname:
            # Discard textbound annotations outside the sentence
            tb_spans = [tb for tb in tb_spans_by_fname[fname]
                    if tb in sentence_tag]

            if argp.debug:
                print >> stderr, 'Enchancing: {}'.format(fname)
                print >> stderr, tb_spans

            for np in (s for s in tag_spans if s.type_ == 'NP'):
                # Get the relevant PoS tags for this NP
                tags = [s for s in tag_spans if s in np]

                if any(s for s in tags if s.type_ == 'CC'):
                    if not argp.no_warn:
                        #print >> stderr, tags
                        print >> stderr, ('WARNING: {}: Skipping NP due to CC'
                                ).format(fname)
                        #raw_input()
                    continue

                #print tags

                # Find the span of the NP head
                #print >> stderr, 'before_head:', tags 
                while any(s for s in tags if s.type_ in ('PP', 'VP', )):
                    # Find the first PP or VP from the right
                    for span_i, span in enumerate(tags[::-1],
                            start=len(tags) - 1):
                        if span.type_ in ('PP', 'VP', ):
                            break
                    else:
                        assert False, 'can not happen (tm)'
                    to_strip = span

                    # Now remove all sub-spans what we removed
                    tags = [s for s in tags
                            if s not in to_strip and s != to_strip]
                    #print >> stderr, 'during_head:', tags 
               
                for span in tags[::-1]:
                    if span.type_ in set(('NN', 'NNS', 'NNP', 'NNPS', )):
                        np_head = span
                        break
                else:
                    # No nouns present, choose the right-most token
                    np_head = tags[-1]

                # Check all annotations contained in the NP
                tb_candidates = []
                in_np_head = [s for s in tb_spans if s in np_head]
                for tb_span in in_np_head:
                    # Restrict the debug output slightly, only in same NP
                    if argp.debug and tb_span in np:
                        print >> stderr, np.text
                        print >> stderr, '{}|{}|'.format(
                                ' ' * (np_head.start - np.start),
                                ' ' * (np_head.end - np_head.start - 2))
                        print >> stderr, '{}|{}|'.format(
                                ' ' * (tb_span.start - np.start),
                                ' ' * (tb_span.end - tb_span.start - 2))
                    # If they cover the head, they can potentially be used
                    if np_head in tb_span:
                        tb_candidates.append(tb_span)
                # Remove duplicate canditates (some annotations aren't uniqued)
                tb_candidates = [s for s in set(tb_candidates)]

                if len(tb_candidates) > 1:
                    if not argp.no_warn: #XXX:
                        print >> stderr, ('WARNING: {}: Skipping NP due to '
                                'multiple TB;s ({}) overlapping the NP head '
                                '({})').format(fname, ', '.join(s.type_
                                    for s in tb_candidates), np_head.text)
                    continue
                    
                if argp.non_alpha_heuristic and any(c for c in np_head.text
                        if not (c.isalpha() or c.isdigit())):
                    alphanum_char_pos = set(i for i, c in enumerate(np_head.text)
                            if c.isalpha() or c.isdigit())

                    # If we have non-alphanum characters and not varying type
                    #   among the annotations for the np head
                    if alphanum_char_pos and len(set(s.type_ for s in in_np_head)) == 1:
                        for tb_span in in_np_head:
                            # Remove any overlapping character
                            alphanum_char_pos = alphanum_char_pos - set(
                                    xrange(tb_span.start - np_head.start,
                                        tb_span.end - np_head.start))

                        # If there are no characters left, we covered the head
                        if not alphanum_char_pos:
                            start = in_np_head[0].start
                            end = in_np_head[-1].end
                            text = np.text[start - np.start:end - np.end]
                            tb_candidates.append(Span(start, end,
                                    type_=in_np_head[0].type_,
                                    text=text,
                                    id_='T{}'.format(next(next_tb_id))))

                if tb_candidates:
                    new_tb_spans = [
                            Span(np.start, np.end,
                                type_=tb_span.type_,
                                text=np.text,
                                id_='T{}'.format(next(next_tb_id)))
                            for tb_span in tb_candidates]

                    if argp.generate:
                        # Generate new spans, dropping pre spans
                        gen_spans = []
                        for new_tb_span in new_tb_spans:
                            tags = [s for s in tag_spans if s in new_tb_span]
                            # Drop the initial np
                            tags = tags[1:]

                            gen_tags = tags
                            while (len(gen_tags) > 1
                                    and gen_tags[0].type_ in PRE_DROPS):
                                gen_tags = gen_tags[1:]
                                first = gen_tags[0]
                                gen_spans.append(Span(first.start, np.end,
                                    type_=tb_span.type_, text=np.text[first.start - np.start:],
                                id_='T{}'.format(next(next_tb_id))))

                        #print >> stderr, 'GENERATED:', gen_spans
                        for s in gen_spans:
                            new_tb_spans.append(s)
                            
                    ann_lines = []
                    for new_tb_span in new_tb_spans:
                        ann_line = '{}\t{} {} {}\t{}'.format(new_tb_span.id_,
                                new_tb_span.type_, new_tb_span.start,
                                new_tb_span.end, new_tb_span.text)
                        ann_lines.append(ann_line)

                        if argp.verbose or argp.debug:
                            #if argp.debug:
                            #    print >> stderr, 'np.text:', np.text
                            print >> stderr, ann_line
                            if argp.debug:
                                raw_input()

                    if not argp.dry_run:
                        out_fname = fname
                        if not argp.merge:
                            out_fname += '.expanded'
                        with open(out_fname, 'a') as expanded_file:
                            expanded_file.write('\n'.join(ann_lines) + '\n')


    return 0

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
