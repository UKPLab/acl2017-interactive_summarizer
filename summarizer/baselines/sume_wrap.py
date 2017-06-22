#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nltk.tokenize import word_tokenize 
import summarizer.baselines.sume as sume
from sume import Sentence, untokenize
from summarizer.algorithms._summarizer import Summarizer
from summarizer.utils.data_helpers import get_parse_info, prune_phrases
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

class SumeWrap(Summarizer):
    def __init__(self, language):
        self.s = sume.ConceptBasedILPSummarizer(" ", language)
        self.LANGUAGE = language
        self.stoplist = set(stopwords.words(self.LANGUAGE))
        self.stemmer = SnowballStemmer(self.LANGUAGE) 
    
    def load_sume_sentences(self, docs, parse_type=None, parse_info=[]):
        """

        :param docs: the documents to load
        :param parse_type:
        :param parse_info:
        :return: list[Sentence]

        @type docs: list[tuple]
        @type parse_type: str
        @type parse_info: list
        """
        doc_sentences = []
        doc_id = 0
        for doc_id, doc in enumerate(docs):
            doc_name, doc_sents = doc
            for sent_pos, sentence in enumerate(doc_sents):
                token_sentence = word_tokenize(sentence, self.LANGUAGE)
                if parse_info:
                    parse_sent = parse_info[0][doc_id][1][sent_pos]
                    hash_tokens_pos, phrases = get_parse_info(parse_sent, self.stemmer, self.LANGUAGE, self.stoplist)
                    pruned_phrases = prune_phrases(phrases, self.stoplist, self.stemmer, self.LANGUAGE)
                    sentence_s = Sentence(token_sentence, doc_id, sent_pos+1, pruned_phrases, hash_tokens_pos)
                else:
                    sentence_s = Sentence(token_sentence, doc_id, sent_pos+1)
                            
                #print token_sentence
                untokenized_form = untokenize(token_sentence)
                sentence_s.untokenized_form = untokenized_form
                sentence_s.length = len(untokenized_form.split(' '))
                doc_sentences.append(sentence_s)
            
        return doc_sentences

    def __call__(self, docs, length=100, units="WORDS", rejected_list=[], imp_list=[], parser_type=None):
        try:
            length = int(length)
        except:
            raise TypeError("argument 'length' could not be converted to int. It is of type '%s' and has value '%s'" % (type(length), length))
        # load documents with extension 'txt'
        self.s.sentences = self.load_sume_sentences(docs, parser_type)
    
        # compute the parameters needed by the model
        # extract bigrams as concepts
        self.s.extract_ngrams2()
    
        # compute document frequency as concept weights
        self.s.compute_document_frequency()
    
        # prune sentences that are shorter than 10 words, identical sentences and
        # those that begin and end with a quotation mark
        if rejected_list:
            self.s.prune_concepts("list", 3, rejected_list)
    
        self.s.prune_sentences(remove_citations=True, remove_redundancy=True, imp_list=imp_list)
    
        # solve the ilp model
        value, subset = self.s.solve_ilp_problem(summary_size=length, units=units)
    
        return [self.s.sentences[j].untokenized_form for j in subset]

