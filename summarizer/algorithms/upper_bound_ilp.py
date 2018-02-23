import pulp
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.data import load as LPickle

import sys, os.path as path
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from summarizer.utils.data_helpers import extract_ngrams2, prune_ngrams, untokenize
from summarizer.algorithms.base import Sentence
from _summarizer import Summarizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

sent_detector = LPickle('tokenizers/punkt/english.pickle')

class ExtractiveUpperbound(Summarizer):
    def __init__(self, language):
        self.sentences = []
        self.docs = []
        self.models = []
        self.doc_sent_dict = {}
        self.ref_ngrams = []
        self.LANGUAGE = language
        self.stemmer = SnowballStemmer(self.LANGUAGE)
        self.stoplist = set(stopwords.words(self.LANGUAGE)) 

    def __call__(self, docs, models, length, ngram_type=2):
        self.sum_length = int(length)
        self.load_data(docs, models)
        self.get_ref_ngrams(ngram_type)
        self.ref_ngrams = prune_ngrams(self.ref_ngrams, self.stoplist, ngram_type)
        #self.prune_sentences(remove_citations=True, remove_redundancy=True)

        self.sentences_idx = range(len(self.sentences))
        self.ref_ngrams_idx = range(len(self.ref_ngrams))

        summary_idx = self.solve_ilp(ngram_type)
        summary_txt = self.get_summary_text(summary_idx)

        return summary_txt

    def load_data(self, docs, models):
        '''
        Load the data into
            :doc_sent_dict
            :sentences

        Parameters:
        docs: List of list of docs each doc is represented with its filename and sents
            [['filename1', ['sent1','sent2','sent3']],['filename2'], ['sent1','sent2','sent3']] ]
        models: List of list of models each doc is represented with its filename and sents
            [['filename1', ['sent1','sent2','sent3']],['filename2'], ['sent1','sent2','sent3']] ]

        '''
        self.docs = docs
        self.models = models
        self.sentences = []
        self.doc_sent_dict = {}

        doc_id = 0
        for doc_id, doc in enumerate(docs):
            _, doc_sents = doc
            total = len(self.sentences)
            for sent_id, sentence in enumerate(doc_sents):
                token_sentence = word_tokenize(sentence, self.LANGUAGE)
                sentence_s = Sentence(token_sentence, doc_id, sent_id+1)

                untokenized_form = untokenize(token_sentence)
                sentence_s.untokenized_form = untokenized_form
                sentence_s.length = len(untokenized_form.split(' '))
                self.doc_sent_dict[total+sent_id] = "%s_%s" % (str(doc_id), str(sent_id))
                self.sentences.append(sentence_s)

    def prune_sentences(self,
                        mininum_sentence_length=5,
                        remove_citations=True,
                        remove_redundancy=True,
                        imp_list=[]):
        """Prune the sentences.

        Remove the sentences that are shorter than a given length, redundant
        sentences and citations from entering the summary.

        Args:
            mininum_sentence_length (int): the minimum number of words for a
              sentence to enter the summary, defaults to 5
            remove_citations (bool): indicates that citations are pruned,
              defaults to True
            remove_redundancy (bool): indicates that redundant sentences are
              pruned, defaults to True
        """
        pruned_sentences = []

        # loop over the sentences
        for i, sentence in enumerate(self.sentences):
            if imp_list:
                if imp_list[i] == 0:
                    continue
            # prune short sentences
            if sentence.length < mininum_sentence_length:
                continue

            # prune citations
            first_token, last_token = sentence.tokens[0], sentence.tokens[-1]
            if remove_citations and \
               (first_token == u"``" or first_token == u'"') and \
               (last_token == u"''" or first_token == u'"'):
                continue

            # prune identical and almost identical sentences
            if remove_redundancy:
                is_redundant = False
                for prev_sentence in pruned_sentences:
                    if sentence.tokens == prev_sentence.tokens:
                        is_redundant = True
                        break

                if is_redundant:
                    continue

            # otherwise add the sentence to the pruned sentence container
            pruned_sentences.append(sentence)

        self.sentences = pruned_sentences


    def get_ref_ngrams(self, N):
        for _, summary in self.models:
            self.ref_ngrams.extend(extract_ngrams2(summary, self.stemmer, self.LANGUAGE, N))

    def get_summary_text(self, summary_idx):
        return [ self.sentences[idx].untokenized_form for idx in summary_idx]

    def solve_ilp(self, N):
        # build the A matrix: a_ij is 1 if j-th gram appears in the i-th sentence

        A = np.zeros((len(self.sentences_idx), len(self.ref_ngrams_idx)))
        for i in self.sentences_idx:
            sent = self.sentences[i].untokenized_form
            sngrams = list(extract_ngrams2([sent], self.stemmer, self.LANGUAGE, N))
            for j in self.ref_ngrams_idx:
                if self.ref_ngrams[j] in sngrams:
                    A[i][j] = 1

        # Define ILP variable, x_i is 1 if sentence i is selected, z_j is 1 if gram j appears in the created summary
        x = pulp.LpVariable.dicts('sentences', self.sentences_idx, lowBound=0, upBound=1, cat=pulp.LpInteger)
        z = pulp.LpVariable.dicts('grams', self.ref_ngrams_idx, lowBound=0, upBound=1, cat=pulp.LpInteger)

        # Define ILP problem, maximum coverage of grams from the reference summaries
        prob = pulp.LpProblem("ExtractiveUpperBound", pulp.LpMaximize)
        prob += pulp.lpSum(z[j] for j in self.ref_ngrams_idx)

        # Define ILP constraints, length constraint and consistency constraint (impose that z_j is 1 if j
        # appears in the created summary)
        prob += pulp.lpSum(x[i] * self.sentences[i].length for i in self.sentences_idx) <= self.sum_length

        for j in self.ref_ngrams_idx:
            prob += pulp.lpSum(A[i][j] * x[i] for i in self.sentences_idx) >= z[j]

        # Solve ILP problem and post-processing to get the summary
        try:
            print('Solving using CPLEX')
            prob.solve(pulp.CPLEX(msg=0))
        except:
            print('Fall back to GLPK')
            prob.solve(pulp.GLPK(msg=0))
                

        summary_idx = []
        for idx in self.sentences_idx:
            if x[idx].value() == 1.0:
                summary_idx.append(idx)

        return summary_idx
