# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals


from collections import namedtuple
from operator import attrgetter
from ..utils import ItemsCount
from .._compat import to_unicode
from ..nlp.stemmers import null_stemmer
from nltk import word_tokenize

SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))


class AbstractSummarizer(object):
    def __init__(self, stemmer=null_stemmer):
        if not callable(stemmer):
            raise ValueError("Stemmer has to be a callable object")

        self._stemmer = stemmer

    def __call__(self, document, sentences_count):
        raise NotImplementedError("This method should be overriden in subclass")

    def stem_word(self, word):
        return self._stemmer(self.normalize_word(word))

    def normalize_word(self, word):
        return to_unicode(word).lower()

    def _get_break_index(self, infos, summary_size):
        s_length = 0
        for i,_ in enumerate(infos):
            s_length += len(word_tokenize("%s" % (infos[i].sentence)))
            if summary_size <= s_length:
                return i+1

    def _get_best_sentences(self, sentences, summary_size, rating, *args, **kwargs):
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        index = self._get_break_index(infos, summary_size)
        # sort sentences by their order in document
        infos = sorted(infos[:index], key=attrgetter("order"))

        return [unicode(i.sentence) for i in infos]
