'''
XXX:

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2011-04-09
'''

from sys import path as sys_path
from os.path import join as path_join
from os.path import dirname
from itertools import chain

from liblinear import LibLinearClassifier

sys_path.append(path_join(dirname(__file__), '..'))

from features import AbstractFeature, SIMPLE_SPAN_INTERNAL_CLASSES
from classifier.simstring.classifier import (SimStringEnsembleFeature,
        SimStringGazetterEnsembleFeature, TsuruokaEnsembleFeature)

class SimpleInternalEnsembleFeature(AbstractFeature): 
    def __init__(self):
        self.features = [c() for c in chain(
            SIMPLE_SPAN_INTERNAL_CLASSES,
            (
                CollinsPatternStringFeature,
                CollapsedCollinsPatternStringFeature,
                FormattingStringFeature,
                ),
            )
            ]

    def get_id(self):
        return 'SIMPLE-INTERNAL-ENSEMBLE'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                #print feature.get_id(), f_tup[0], f_tup[1]
                yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])
        #assert False


from features import (
        # Single
        CollinsPatternStringFeature,
        CollapsedCollinsPatternStringFeature,
        FormattingStringFeature,

        # Sentence
        SentenceCollinsPatternStringFeature,
        SentenceCollapsedCollinsPatternStringFeature,
        SentenceLowerCaseStringFeature,
        SentencePorterStemStringFeature,

        # Window
        WindowStringFeature,
        WindowLowerCaseStringFeature,
        WindowPrefixStringFeature,
        WindowSuffixStringFeature,
        WindowPorterStemStringFeature,
        WindowCollinsPatternStringFeature,
        WindowCollapsedCollinsPatternStringFeature,
        WindowFormattingStringFeature,
        )


class CompetitiveEnsembleFeature(AbstractFeature):
    def __init__(self):
        self.features = [c() for c in chain(
            SIMPLE_SPAN_INTERNAL_CLASSES,
            
            # Single classes
            (
            # Single token internal
            CollinsPatternStringFeature,
            CollapsedCollinsPatternStringFeature,
            FormattingStringFeature,
            
            # Sentence level
            #SentenceCollinsPatternStringFeature,
            #SentenceCollapsedCollinsPatternStringFeature,
            #SentenceLowerCaseStringFeature,
            #SentencePorterStemStringFeature,

            # Word window level
            WindowStringFeature,
            #WindowLowerCaseStringFeature,
            #WindowPrefixStringFeature,
            #WindowSuffixStringFeature,
            #WindowPorterStemStringFeature,
            #WindowCollinsPatternStringFeature,
            #WindowCollapsedCollinsPatternStringFeature,
            #WindowFormattingStringFeature,

            )
            )]

    def get_id(self):
        return 'COMPETITIVE-ENSEMBLE'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                #print feature.get_id(), f_tup[0], f_tup[1]
                yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])
        #assert False


'''
class SimStringCompetitiveEnsembleFeature(CompetitiveEnsembleFeature):
    def __init__(self):
        CompetitiveEnsembleFeature.__init__(self)
        self.features.add(SimStringGazetterEnsembleFeature)
'''

class SimpleInternalEnsembleClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = SimpleInternalEnsembleFeature
        

class CompetitiveEnsembleClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = CompetitiveEnsembleFeature

DONT_FILTER_TURKU = True
class SimStringCompetitiveEnsembleFeature(AbstractFeature):
    def __init__(self):
        self.features = [c() for c in [
            #CompetitiveEnsembleFeature,
            #SimpleInternalEnsembleFeature,

            # SimString Features
            SimStringEnsembleFeature,
            #SimStringGazetterEnsembleFeature
            # TODO: Contextual SimString
            ]]

    def get_id(self):
        return 'SIMSTRING-COMPETITIVE-ENSEMBLE'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                if DONT_FILTER_TURKU or 'turku' not in f_tup[0]:
                    yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])

class SimStringCompetitiveEnsembleClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = SimStringCompetitiveEnsembleFeature

class InternalFeature(AbstractFeature):
    def __init__(self):
        self.features = [c() for c in [
            #CompetitiveEnsembleFeature,
            SimpleInternalEnsembleFeature,

            # SimString Features
            #SimStringEnsembleFeature,
            #SimStringGazetterEnsembleFeature
            # TODO: Contextual SimString
            ]]

    def get_id(self):
        return 'INTERNAL'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])


class InternalClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = InternalFeature


class SimStringInternalFeature(AbstractFeature):
    def __init__(self):
        self.features = [c() for c in [
            #CompetitiveEnsembleFeature,
            SimpleInternalEnsembleFeature,
            #XXX: XXX: HACK! REMOVE!
            #WindowStringFeature,

            # SimString Features
            SimStringEnsembleFeature,
            #SimStringGazetterEnsembleFeature
            # TODO: Contextual SimString
            ]]

    def get_id(self):
        return 'SIMSTRING-INTERNAL'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                if DONT_FILTER_TURKU or 'turku' not in f_tup[0]:
                    yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])


class SimStringInternalClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = SimStringInternalFeature

class SimStringTsuruokaFeature(AbstractFeature):
    def __init__(self):
        self.features = [c() for c in [
            #CompetitiveEnsembleFeature,

            # SimString Features
            TsuruokaEnsembleFeature,
            SimStringEnsembleFeature,
            #SimStringGazetterEnsembleFeature
            # TODO: Contextual SimString
            ]]

    def get_id(self):
        return 'SIMSTRING-TSURUOKA'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                if DONT_FILTER_TURKU or 'turku' not in f_tup[0]:
                    yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])

class SimStringTsuruokaInternalFeature(AbstractFeature):
    def __init__(self):
        self.features = [c() for c in [
            #CompetitiveEnsembleFeature,
            SimpleInternalEnsembleFeature,

            # SimString Features
            TsuruokaEnsembleFeature,
            SimStringEnsembleFeature,
            #SimStringGazetterEnsembleFeature
            # TODO: Contextual SimString
            ]]

    def get_id(self):
        return 'SIMSTRING-TSURUOKA-INTERNAL'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                if DONT_FILTER_TURKU or 'turku' not in f_tup[0]:
                    yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])

class SimStringTsuruokaInternalClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = SimStringTsuruokaInternalFeature

class SimStringTsuruokaClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = SimStringTsuruokaFeature

class TsuruokaInternalFeature(AbstractFeature):
    def __init__(self):
        self.features = [c() for c in [
            #CompetitiveEnsembleFeature,
            SimpleInternalEnsembleFeature,

            # SimString Features
            TsuruokaEnsembleFeature,
            #SimStringGazetterEnsembleFeature
            # TODO: Contextual SimString
            ]]

    def get_id(self):
        return 'TSURUOKA-INTERNAL'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                if DONT_FILTER_TURKU or 'turku' not in f_tup[0]:
                    yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])

class TsuruokaInternalClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = TsuruokaInternalFeature

class TsuruokaClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = TsuruokaEnsembleFeature

class GazetterInternalFeature(AbstractFeature):
    def __init__(self):
        self.features = [c() for c in [
            #CompetitiveEnsembleFeature,
            SimpleInternalEnsembleFeature,

            # SimString Features
            #SimStringEnsembleFeature,
            SimStringGazetterEnsembleFeature
            # TODO: Contextual SimString
            ]]

    def get_id(self):
        return 'SIMSTRING-GAZETTER'

    def featurise(self, document, sentence, annotation):
        for feature in self.features:
            for f_tup in feature.featurise(document, sentence, annotation):
                if DONT_FILTER_TURKU or 'turku' not in f_tup[0]:
                    yield (f_tup[0] + '-(<' + feature.get_id() + '>)', f_tup[1])
     
        
class GazetterInternalClassifier(LibLinearClassifier):
    def __init__(self):
        LibLinearClassifier.__init__(self)
        self.feature_class = GazetterInternalFeature
