'''
Naive string matching and majority voting classifier.

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-02-28
'''

from random import choice
from collections import defaultdict

#try:
#    from cPickle import dump, load, HIGHEST_PROTOCOL
#except ImportError:
#    from pickle import dump, load, HIGHEST_PROTOCOL


### TODO: This belongs in its own module
class Classifier(object):
    def classify(self, document, sentence, annotation):
        raise NotImplementedError
    
    def train(self, documents):
        raise NotImplementedError


class CouldNotClassifyError(Exception):
    pass


class ClassifierNotTrainedError(Exception):
    pass
###


# Just selects the most frequent class, always
class MaximumClassifier(Classifier):
    def __init__(self):
        self.count_by_class = None

    def classify(self, document, sentence, annotation):
        if self.count_by_class is None:
            raise ClassifierNotTrainedError
       
        # 
        max_count = max((count for count in self.count_by_class.itervalues()))

        most_likely = [_class for _class, count
                    in self.count_by_class.iteritems()
                    if count == max_count]
           
        # If there were multiple we will pick one at random
        return choice(most_likely)
        
    def train(self, documents):
        if self.count_by_class is None:
            self.count_by_class = {}

        # Iterate over all documents, sentences and annotations
        for document in documents:
            for sentence in document:
                for annotation in sentence:
                    try:
                        self.count_by_class[annotation.type] += 1
                    except KeyError:
                        self.count_by_class[annotation.type] = 0


# Memorise seen instances, if not in memory, fall back to maximum class
class NaiveClassifier(Classifier):
    def __init__(self):
        # We will not use defaultdicts for the sake of pickling
        self.prob_by_class = {}
        self.seen_by_class = {}

    def _classify(self, vec):
        ann_text = vec[0]

        # Try to find a previous observation in the training data
        alternatives = [(_class, self.prob_by_class[_class])
                for _class, seen in self.seen_by_class.iteritems()
                if ann_text in seen]
            
        # Do we have a single clear alternative? If not, me need to rank them
        if len(alternatives) == 1:
            return alternatives[0][0]
        else:
            # Get the maximum probabilites for the alternatives or all classes
            max_prob = max((prob for _, prob
                    in (alternatives or self.prob_by_class.iteritems())))

            most_likely = [_class for _class, prob
                    in (alternatives or self.prob_by_class.iteritems())
                    if prob == max_prob]
           
            # If there were multiple we will pick one at random
            return choice(most_likely)


    def classify(self, document, sentence, annotation):
        return self._classify((sentence.annotation_text(annotation), ))
        '''
        ann_text = sentence.annotation_text(annotation)

        # Try to find a previous observation in the training data
        alternatives = [(_class, self.prob_by_class[_class])
                for _class, seen in self.seen_by_class.iteritems()
                if ann_text in seen]
            
        # Do we have a single clear alternative? If not, me need to rank them
        if len(alternatives) == 1:
            return alternatives[0][0]
        else:
            # Get the maximum probabilites for the alternatives or all classes
            max_prob = max((prob for _, prob
                    in (alternatives or self.prob_by_class.iteritems())))

            most_likely = [_class for _class, prob
                    in (alternatives or self.prob_by_class.iteritems())
                    if prob == max_prob]
           
            # If there were multiple we will pick one at random
            return choice(most_likely)
        '''

    def _gen_lbls_vecs(self, documents):
        lbls = []
        vecs = []

        # Iterate over all documents, sentences and annotations
        for document in documents:
            for sentence in document:
                for annotation in sentence:
                    lbls.append(annotation.type)
                    ann_text = sentence.annotation_text(annotation)
                    vecs.append((ann_text, ))
        return lbls, vecs

    def _train(self, lbls, vecs):
        self.prob_by_class = {}
        self.seen_by_class = {}
        
        from itertools import izip
       
        for lbl, vec in izip(lbls, vecs):
            try:
                self.seen_by_class[lbl]
            except KeyError:
                self.seen_by_class[lbl] = {}

            ann_text = vec[0]
            try:
                self.seen_by_class[lbl][ann_text] += 1
            except KeyError:
                self.seen_by_class[lbl][ann_text] = 1

        # Calculate the maximum likelihood for each class based on occurences
        total_seen = float(sum((sum(seen.itervalues())
            for seen in self.seen_by_class.itervalues())))
        for _class, seen in self.seen_by_class.iteritems():
            self.prob_by_class[_class] = sum(seen.itervalues()) / total_seen

    def train(self, documents):
        lbls, vecs = self._gen_lbls_vecs(documents)
        self._train(lbls, vecs)
        '''
        return
        # Iterate over all documents, sentences and annotations
        for document in documents:
            for sentence in document:
                for annotation in sentence:
                    try:
                        self.seen_by_class[annotation.type]
                    except KeyError:
                        self.seen_by_class[annotation.type] = {}

                    ann_text = sentence.annotation_text(annotation)
                    try:
                        self.seen_by_class[annotation.type][ann_text] += 1
                    except KeyError:
                        self.seen_by_class[annotation.type][ann_text] = 1

        # Calculate the maximum likelihood for each class based on occurences
        total_seen = float(sum((sum(seen.itervalues())
            for seen in self.seen_by_class.itervalues())))
        if self.prob_by_class is None:
            self.prob_by_class = {}
        for _class, seen in self.seen_by_class.iteritems():
            self.prob_by_class[_class] = sum(seen.itervalues()) / total_seen
        #print self.prob_by_class
        '''

def main(args):
    return -1

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
