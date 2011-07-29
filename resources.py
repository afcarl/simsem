'''
Generic resource containers used by the SimSem.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-02-20
'''

# TODO: Do we need unittests for this?

from itertools import chain

### Constants
###

#TODO: Should have a compare so that we can sort them by span
#XXX: We should have a span as well for the times when it is not annotated
#XXX: Can't do across sentences, but that should be fine
class Annotation(object):
    def __init__(self, start, end, type):
        self.start = start
        self.end = end
        self.type = type

    def __str__(self):
        return 'Annotation(type={}, start={}, end={})'.format(
                self.type, self.start, self.end)


class Sentence(object):
    def __init__(self, text, annotations):
        self.text = text
        self.annotations = annotations

    def __getitem__(self, val):
        return self.text.__getitem__(val)

    def __iter__(self):
        return iter(self.annotations)

    def __str__(self):
        return self.text

    def add_annotation(self, annotation):
        self.annotations.append(annotation)

    def annotation_text(self, annotation):
        ret = self.text[annotation.start:annotation.end]
        assert ret, '{} len({}) "{}"'.format(annotation,
                len(self.text), self.text)
        return ret


class Document(object):
    def __init__(self, id, title, abstract, path):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.path = path

    def __iter__(self):
        return chain(self.title, self.abstract)

    def __len__(self):
        return len(self.title) + len(self.abstract)
