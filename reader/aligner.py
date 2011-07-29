'''
Module for aligning textual output.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-03-03
'''


class MisalignedError(Exception):
    pass


class Aligner(object):
    #XXX: We only accept ignores INTERNALLY!
    def __init__(self, target, ignore=None, ignore_mult=None):
        if ignore is None:
            ignore = set()
        if ignore_mult is None:
            ignore_mult = set()

        self.target = target
        self.ignore = ignore
        self.ignore_mult = ignore_mult

        self.current_idx = 0
        self.seen = []
        self.previous = None
        self.char_cnt = 0

    def align(self, char):
        self.char_cnt += 1
        if char == self.target[self.current_idx]:
            self.current_idx += 1
        elif (self.seen and (char in self.ignore
            or (char == self.previous and char in self.ignore_mult))):
            # We accept these exception, but does not align ahead
            pass
        else:
            # It is a mismatch, reset the alignment
            self.current_idx = 0
            self.seen = []
            raise MisalignedError

        self.seen.append(char)
        self.previous = char

        # Are we done?
        return self.current_idx >= len(self.target)

    # Gives a string of what we have seen, not necessarily the target
    def __repr__(self):
        return 'Aligner<seen="{}" to_see="{}">'.format(
                ''.join(self.seen), self.target)

    def __str__(self):
        return ''.join(self.seen)


if __name__ == '__main__':
    # TODO: Turn this into a unittest
    source = ('Administration s.c. is not inferior to i.v.  '
            'Side effects mainly consist of flu-like symptoms and headache. '
            'The role of rhIL-3 after high-dose chemotherapy and autologous '
            'bone marrow reinfusion is still questionable')
    target = ('is not inferior to i.v. '
            'Side effects mainly consist of flu-like symptoms and headache.')
    aligner = Aligner(target, ignore_mult=set((' ',)))

    start_idx = 0
    for idx, char in enumerate(source):
        try:
            if aligner.align(char):
                s_aligned = str(aligner)
                break
        except MisalignedError:
            start_idx = idx + 1

    s_source = source[start_idx:start_idx + len(s_aligned)]
    assert s_aligned == s_source, ('difference between aligned and source '
            's_aligned: "{}" !=  s_source "{}"'
            ).format(s_aligned, s_source)
