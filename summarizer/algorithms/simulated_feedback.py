from __future__ import print_function
import json
import itertools
import re

import os.path as path
import sys

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import tempfile
import random

import numpy as np
import pulp
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn import svm

from summarizer.algorithms.feedback_graph import SimpleNgramFeedbackGraph, PageRankFeedbackGraph
from summarizer.algorithms.flight_recorder import FlightRecorder, Record
from summarizer.baselines import sume
from summarizer.baselines.sume_wrap import SumeWrap
from summarizer.utils.data_helpers import prune_ngrams, extract_ngrams2, get_parse_info, \
    prune_phrases


RECOMMENDER_METHOD_SAMPLING = "SAMPLING"
RECOMMENDER_METHOD_HIGHEST_WEIGHT = "HIGHEST_WEIGHT"

PARSE_TYPE_PARSE = 'parse'
ORACLE_TYPE_CUSTOM_WEIGHT = 'CUSTOM_WEIGHT'
ORACLE_TYPE_TOP_N = 'top_n'
ORACLE_TYPE_KEEPTRACK = 'keeptrack'
ORACLE_TYPE_ACTIVE_LEARNING = 'active_learning'
ORACLE_TYPE_ACTIVE_LEARNING2 = 'active_learning2'
ORACLE_TYPE_ILP_FEEDBACK = "ilp_feedback"
ORACLE_TYPE_ACCEPT_REJECT = 'accept_reject'
ORACLE_TYPE_ACCEPT = 'accept'
ORACLE_TYPE_REJECT = 'reject'
ORACLE_TYPE_ACCEPT_ALL = 'accept_all'
ORACLE_TYPE_REJECT_ALL = 'reject_all'

CHANGE_WEIGHT_MODE_ACCEPT = 'accept'
CHANGE_WEIGHT_MODE_REJECT = 'reject'
CHANGE_WEIGHT_MODE_IMPLICIT_REJECT = 'implicit_reject'


from summarizer.utils.writer import write_to_file

class Oracle():
    def reject_concepts(self, summ_concepts, ref_concepts):
        '''
        Reject Ngrams

        Keyword arguments:
        ref_ngrams: list of reference n-gram tuples
                    ['1 2', '2 3', '3 4']
        summ_ngrams: list of summary n-gram tuples
                    ['1 2', '2 4']

        return:
        Return N-grams not present in reference
                    ['2 4']
        '''
        return set(summ_concepts) - ref_concepts

    def accept_concepts(self, summ_concepts, ref_concepts):
        '''
        Accept Ngrams

        Keyword arguments:
        ref_ngrams: list of reference n-gram tuples
                    ['1 2', '2 3', '3 4']
        summ_ngrams: list of summary n-gram tuples
                    ['1 2', '2 4']

        return: Overlap of N-grams
                    ['1 2']

        '''
        return set(summ_concepts) & ref_concepts


class SimulatedFeedback(object):
    def __init__(self, language, rouge, embeddings={}, fvector=[], ngrams_size=2, top_n=100, dump_base_dir=tempfile.mkdtemp(prefix="simufee-")):
        '''
        Initialize the docs and models structure
        '''
        self.Oracle = Oracle()  # oracle
        self.SumeWrap = SumeWrap(language) # only used to load the sentences and push them into self.summarizer
        self.summarizer = sume.ConceptBasedILPSummarizer(" ", language)
        self.N = ngrams_size # how many words an should the ngrams consist of
        self.top_n = top_n  # currently unused
        self.ref_ngrams = set() # set of ngrams that are in the reference summaries (for the feedback to peek)
        self.ref_phrases = set() # set of phrases that are in the reference summaries (for the feedback to peek)

        self.flight_recorder = FlightRecorder()  # The flight-recorder stores all interactions wrt to concepts (eg. accepted, and rejected)

        self.info_data = [] # stats for the pipeline. The only thing that leaves this class
        self.initial_weights = {} # oracle reweighting
        self.language = language # document language. relevant for stemmer, embeddings, stopwords, parsing
        #self.stemmer = SnowballStemmer(self.language)
        if self.language == "english":
            self.stemmer = SnowballStemmer(self.language)
            #elf.stemmer = WordNetLemmatizer()
        else:
            self.stemmer = SnowballStemmer(self.language)
        self.stoplist = set(stopwords.words(self.language))
        self.rouge = rouge
        self.cluster_size = 0.0
        self.embeddings = embeddings # word2vec embeddings
        self.fvector = fvector # List of support vectors for active learning SVM
        self.pos_hash = {} # active learning // SVM
        self.concept_vec_idx = {} # active learning // SVM
        self.index_vec_concept = {} # active learning // SVM

        ### previously uninitialized fields...
        self.data = None # np.array(self.fvector)   # active learning // SVM TODO rename self.data to somehting that contains svm...
        self.labels = None # active learning // SVM
        self.MAX_WEIGHT = None # int with # of documents (i.e. largest possible DF value)
        self.models = None # reference summaries, only needed for rouge score (as they are converted merged into one large summary)
        self.parse_type = None # None or "parse"
        self.prev_score = None # rouge scores of previous iteration.
        self.score = None # rouge scores of current iteration.
        self.summary_length = None # target summary length.
        self.ub_score = None # rouge scores of upper bound
        self.uncertainity = {} # active learning // SVM

        # graph based propagation settings
        self.graph = PageRankFeedbackGraph(self.stemmer, self.language)
        # self.graph = SimpleNgramFeedbackGraph(self.stemmer, self.language, N=5)
        self.debug_dump_target_dir = dump_base_dir
        self.allowed_number_of_feedback_per_iteration=5

    def get_sorted_concepts(self):
        '''
        Get sorted concepts
        '''
        sorted_concepts = sorted(self.summarizer.weights,
                                 key=lambda x: self.summarizer.weights[x],
                                 reverse=True)

        # iterates over the concept weights
        return sorted_concepts

    def get_implicit_feedback(self, summ_ngrams, list_concepts):
        feedback_keys = []
        for key in summ_ngrams:
            for phrase in list_concepts:
                if re.search(u'(\s|^)%s([\s]|$)' % (key), u'%s' % (phrase)) or re.search(u'(\s|^)%s([\s]|$)' % (phrase),
                                                                                         u'%s' % (key)):
                    # print(key, phrase)
                    feedback_keys.append(key)
        implicit_feedback = set(summ_ngrams) - set(feedback_keys)
        return implicit_feedback

    def get_feedback(self, subset, recommender=None):
        """
            Generate feedback for the subset sentences by peeking into the reference summary.

        :param subset: The indices of the sentences to get feedback for.
        :param allowed_number_of_feedbacks: how many concepts may be sent to the oracle, default all
        """
        new_implicit_rejects = set() # currently not used (all writing occurences are commented out)

        summary = [self.summarizer.sentences[j].untokenized_form for j in subset]
        # print('Feedback-optimal summary:', summary)

        if self.parse_type == 'parse':
            print('feedback on phrases')
            summary_phrases = [self.summarizer.sentences[j].phrases for j in subset]

            samples = list(itertools.chain(*summary_phrases))
            references=self.ref_phrases

        elif self.parse_type == None:
            print('feedback on ngrams')
            summary_concepts = [self.summarizer.sentences[j].concepts for j in subset]
            
            samples = list(itertools.chain(*summary_concepts))
            references = self.ref_ngrams

        # from all samples, use a sub-set
        if recommender is None:
            use_samples = samples
        elif recommender == RECOMMENDER_METHOD_SAMPLING:
            use_samples = random.sample(samples, self.allowed_number_of_feedback_per_iteration)
        elif recommender == RECOMMENDER_METHOD_HIGHEST_WEIGHT:
            use_samples = self.recommend_highest_weight(samples, self.allowed_number_of_feedback_per_iteration);

        new_rejects = list(self.Oracle.reject_concepts(use_samples, references) - self.flight_recorder.union().reject)
        new_accepts = list(self.Oracle.accept_concepts(use_samples, references) - self.flight_recorder.union().accept)

        new_rejects = prune_ngrams(new_rejects, self.stoplist, self.N)
        new_accepts = prune_ngrams(new_accepts, self.stoplist, self.N)


        '''
        if self.parse_type == 'parse':
            self.recorder.total_accept_keys += self.project_phrase_ngrams(self.recorder.accepted_concepts)
            self.recorder.total_reject_keys += self.project_phrase_ngrams(self.recorder.rejected_concepts)
            
            x = list(Set(self.recorder.total_accept + self.recorder.union.reject))
            new_implicit_rejects = list(self.get_implicit_feedback(summ_ngrams, x) - Set(self.recorder.total_implicit_reject))
            # self.recorder.total_implicit_reject += self.recorder.latest().implicit_reject
        '''

        # self.recorder.total_accept += self.recorder.accepted_concepts
        # self.recorder.total_reject += self.recorder.rejected_concepts
        # self.recorder.total_implicit_reject += self.recorder.latest().implicit_reject
        return (new_accepts, new_rejects, new_implicit_rejects)

    def recommend_highest_weight(self, samples, limit=1, prune=True):
        w = dict(self.graph.get_weights())
        s = sorted(w, key=w.get, reverse=True)
        s = [item for item in s if 0.0 < w.get(item) < 1.0
                 and item not in self.flight_recorder.union().reject
                 and item not in self.flight_recorder.union().accept
                 and item not in self.flight_recorder.union().implicit_reject]

        pruned = prune_ngrams(s, self.stoplist, self.N)
        result =[]
        for concept in s:
            if concept in samples:
                # print ("adding %s with weight %s to result" % (concept, w[concept]))
                result.append(concept)

        return result[:limit]



    def partial_feedback(self, ngrams_list):
        return [ngram for ngram in ngrams_list if self.summarizer.weights[ngram] > 1]

    def change_weights(self, concept_list, oracle_type):
        for key in concept_list:
            if oracle_type == CHANGE_WEIGHT_MODE_REJECT:
                self.summarizer.weights[key] = 0.0
            if oracle_type == CHANGE_WEIGHT_MODE_ACCEPT:
                self.summarizer.weights[key] = self.MAX_WEIGHT

    def recalculate_weights(self, oracle_type, propagation=False):
        """
            Set new weights in self.summarizer.weights according to the currently selected feedbcack method.

            This method basically interprets the feedback. if propagation is False, its using the default model, which
            is changing weights based on the FlightRecorder feedback. If propagation is True, changing weights is based
            on graph traversal...

        :param oracle_type:
        """
        # if the graph exists, we need to update it using EXACTLY the same data as the other oracles used.
        if propagation is False:
            self.__update_summarizer_weights_baseline__(oracle_type)
        elif propagation is True:
            self.graph.incorporate_feedback(self.flight_recorder)
            self.__update_summarizer_weights_using_graph__(oracle_type);
            # change weights using the feedbackgraph
        else:
            print("recalculate weights is broken!");

    def __update_summarizer_weights_using_graph__(self, oracle_type=""):
        """

        """
        if self.graph is None:
            raise StandardError("Set to propagation, but no coocurrence_graph is given")

        G = self.graph

        weights = self.summarizer.weights
        for (concept, weight) in G.get_weights():
            if weights.has_key(concept):
                weights[concept] = weight * self.MAX_WEIGHT
            elif weight > 1:
                print("ignoring unknown key: " , concept, " with weight ", weight)

    def __update_summarizer_weights_baseline__(self, oracle_type):
        """
            The original method to update weights: Rejected concepts get weight ZERO, Accepted concepts get weight ONE.

            :param oracle_type:
            :return:
        """
        # self.summarizer.weights = __convert_graph_to_weights__()
        if oracle_type == ORACLE_TYPE_REJECT_ALL:
            self.change_weights(self.flight_recorder.union().reject, CHANGE_WEIGHT_MODE_REJECT)
            if self.parse_type == PARSE_TYPE_PARSE:
                self.change_weights(self.flight_recorder.union().implicit_reject, CHANGE_WEIGHT_MODE_REJECT)
        if oracle_type == ORACLE_TYPE_ACCEPT_ALL:
            self.change_weights(self.flight_recorder.union().accept, CHANGE_WEIGHT_MODE_ACCEPT)
        if oracle_type == ORACLE_TYPE_ACCEPT_REJECT \
                or oracle_type == ORACLE_TYPE_ILP_FEEDBACK \
                or oracle_type.startswith(ORACLE_TYPE_ACTIVE_LEARNING):
            if self.parse_type == None:
                print('Weight change', oracle_type)
                self.change_weights(self.flight_recorder.latest().reject, CHANGE_WEIGHT_MODE_REJECT)
                self.change_weights(self.flight_recorder.latest().accept, CHANGE_WEIGHT_MODE_ACCEPT)
            if self.parse_type == PARSE_TYPE_PARSE:
                self.change_weights(self.project_phrase_ngrams(self.flight_recorder.latest().reject),
                                    CHANGE_WEIGHT_MODE_REJECT)
                self.change_weights(self.project_phrase_ngrams(self.flight_recorder.latest().accept),
                                    CHANGE_WEIGHT_MODE_ACCEPT)
                self.change_weights(self.flight_recorder.latest().implicit_reject, CHANGE_WEIGHT_MODE_REJECT)
        if oracle_type == ORACLE_TYPE_KEEPTRACK:
            if self.parse_type == None:
                self.change_weights(self.flight_recorder.latest().reject, CHANGE_WEIGHT_MODE_REJECT)
                if self.flight_recorder.latest().accept:
                    self.change_weights(self.flight_recorder.union().accept, CHANGE_WEIGHT_MODE_REJECT)
                else:
                    self.change_weights(self.flight_recorder.union().accept, CHANGE_WEIGHT_MODE_ACCEPT)
            if self.parse_type == PARSE_TYPE_PARSE:
                self.change_weights(self.project_phrase_ngrams(self.flight_recorder.latest().reject),
                                    CHANGE_WEIGHT_MODE_REJECT)
                self.change_weights(self.flight_recorder.latest().implicit_reject, CHANGE_WEIGHT_MODE_REJECT)
                if self.flight_recorder.latest().accept:
                    self.change_weights(self.project_phrase_ngrams(self.flight_recorder.latest().accept),
                                        CHANGE_WEIGHT_MODE_ACCEPT)
                else:
                    self.change_weights(self.project_phrase_ngrams(self.flight_recorder.union().accept),
                                        CHANGE_WEIGHT_MODE_ACCEPT)
        if oracle_type == ORACLE_TYPE_TOP_N:
            self.summarizer.weights = self.initial_weights
            self.change_weights(self.flight_recorder.union().reject, CHANGE_WEIGHT_MODE_REJECT)
            self.change_weights(self.flight_recorder.union().accept, CHANGE_WEIGHT_MODE_ACCEPT)
            if self.flight_recorder.union().accept:
                sorted_weights = self.get_sorted_concepts()
                for key in self.summarizer.weights:
                    if key not in sorted_weights[:400]:
                        self.summarizer.weights[key] = 0

    def get_details(self, iteration, summary_length, oracle_type):
        """
            Get details about an ilp iteration. It does actually recalc the weights, solve the ilp, extract the
            relevant information, and resets the weights to the previous value.
        :param iteration:
        :param summary_length:
        :param oracle_type:
        :return:
        """
        
        print("flight rec: (T: %s = A: %s + R: %s ), (L: %s = A: %s + R: %s)" %
              (len(self.flight_recorder.union().accept | self.flight_recorder.union().reject),
               len(self.flight_recorder.union().accept),
               len(self.flight_recorder.union().reject),
               len(self.flight_recorder.latest().accept | self.flight_recorder.latest().reject),
               len(self.flight_recorder.latest().accept),
               len(self.flight_recorder.latest().reject)))
        # solve the ilp model
        value, subset = self.summarizer.solve_ilp_problem(summary_size=int(summary_length), units="WORDS")
        summary = [self.summarizer.sentences[j].untokenized_form for j in subset]

        summary_text = '\n'.join(summary)
        score = self.rouge(summary_text, self.models, self.summary_length)

        accepted = self.flight_recorder.latest().accept
        rejected = self.flight_recorder.latest().reject
        row = [str(iteration), score[0], score[1], score[2], len(accepted), len(rejected),
               summary_text]

        #self.summarizer.weights = old_weights

        print(row[:-1])
        # print(summary_text.encode('utf-8'))
        self.info_data.append(row)
        return summary, score, subset

    def check_break_condition(self, iteration, prev_summary, summary, ub_summary, prev_score):
        if not self.flight_recorder.latest().accept and not self.flight_recorder.latest().reject:
            print("BREAKING HERE: Stopping because last flight_recorder is basically empty")
            return 1
        if self.score[1] >= self.ub_score[1]:  # ROUGE2 score> Upper-bound
            print("BREAKING HERE: current summary is BETTER than UB")
            return 1
        if summary == ub_summary:
            print("BREAKING HERE: Found UB summary")
            return 1
        if self.ub_score == self.score:
            print("BREAKING HERE: score is equal to UB score")
            return 1
        return 0

    def solve_joint_ilp(self, summary_size, feedback, non_feedback, uncertainity={}, labels={}, unique=False, solver='glpk', excluded_solutions=[]):
        """

        :param summary_size: The size of the backpack. i.e. how many words are allowed in the summary.
        :param feedback:
        :param non_feedback:
        :param unique: if True, an boudin_2015 eq. (5) is applied to enforce a unique solution.
        :param solver: cplex, if fails use the mentioned solver
        :param excluded_solutions:
        
        :return: (val, set) tuple (int, list): the value of the objective function and the set of
            selected sentences as a tuple. 
        
        """
        w = self.summarizer.weights
        u = uncertainity
        L = summary_size
        NF = len(non_feedback)
        F = len(feedback)
        S = len(self.summarizer.sentences)

        if not self.summarizer.word_frequencies:
            self.summarizer.compute_word_frequency()

        tokens = self.summarizer.word_frequencies.keys()
        f = self.summarizer.word_frequencies
        T = len(tokens)

        # HACK Sort keys
        # concepts = sorted(self.weights, key=self.weights.get, reverse=True)

        # formulation of the ILP problem
        prob = pulp.LpProblem(self.summarizer.input_directory, pulp.LpMaximize)

        # initialize the concepts binary variables
        nf = pulp.LpVariable.dicts(name='nf',
                                   indexs=range(NF),
                                   lowBound=0,
                                   upBound=1,
                                   cat='Integer')

        f = pulp.LpVariable.dicts(name='F',
                                  indexs=range(F),
                                  lowBound=0,
                                  upBound=1,
                                  cat='Integer')

        # initialize the sentences binary variables
        s = pulp.LpVariable.dicts(name='s',
                                  indexs=range(S),
                                  lowBound=0,
                                  upBound=1,
                                  cat='Integer')

        # initialize the word binary variables
        t = pulp.LpVariable.dicts(name='t',
                                  indexs=range(T),
                                  lowBound=0,
                                  upBound=1,
                                  cat='Integer')

        # OBJECTIVE FUNCTION
        if labels:
            print('solve for Active learning 2')
            prob += pulp.lpSum(w[non_feedback[i]] * (1.0 - u[non_feedback[i]]) * labels[non_feedback[i]] * nf[i] for i in range(NF))
        if not labels:
            if uncertainity:
                print('solve for Active learning')
                if feedback:
                    # In this phase, we force new concepts to be chosen, and not those we already have feedback on, and
                    # therefore non_feedback is added while feedback is substracted from the problem. I.e. by
                    # substracting the feedback, those sentences will disappear from the solution.
                    prob += pulp.lpSum(w[non_feedback[i]] * u[non_feedback[i]] * nf[i] for i in range(NF)) - pulp.lpSum(
                            w[feedback[i]] * u[feedback[i]] * f[i] for i in range(F))
                    pulp.l
                else:
                    prob += pulp.lpSum(w[non_feedback[i]] * u[non_feedback[i]] * nf[i] for i in range(NF))
            if not uncertainity:
                print('solve for ILP feedback')
                if feedback:
                    prob += pulp.lpSum(w[non_feedback[i]] * nf[i] for i in range(NF)) - pulp.lpSum(w[feedback[i]] * f[i] for i in range(F))
                else:
                    prob += pulp.lpSum(w[non_feedback[i]] * nf[i] for i in range(NF))

        if unique:
            prob += pulp.lpSum(w[non_feedback[i]] * nf[i] for i in range(NF)) - pulp.lpSum(w[feedback[i]] * f[i] for i in range(F)) + \
                    10e-6 * pulp.lpSum(f[tokens[k]] * t[k] for k in range(T))

        # CONSTRAINT FOR SUMMARY SIZE
        prob += pulp.lpSum(s[j] * self.summarizer.sentences[j].length for j in range(S)) <= L

        # INTEGRITY CONSTRAINTS
        for i in range(NF):
            for j in range(S):
                if non_feedback[i] in self.summarizer.sentences[j].concepts:
                    prob += s[j] <= nf[i]

        for i in range(NF):
            prob += pulp.lpSum(s[j] for j in range(S)
                        if non_feedback[i] in self.summarizer.sentences[j].concepts) >= nf[i]

        for i in range(F):
            for j in range(S):
                if feedback[i] in self.summarizer.sentences[j].concepts:
                    prob += s[j] <= f[i]

        for i in range(F):
            prob += pulp.lpSum(s[j] for j in range(S)
                        if feedback[i] in self.summarizer.sentences[j].concepts) >= f[i]

        # WORD INTEGRITY CONSTRAINTS
        if unique:
            for k in range(T):
                for j in self.summarizer.w2s[tokens[k]]:
                    prob += s[j] <= t[k]

            for k in range(T):
                prob += pulp.lpSum(s[j] for j in self.summarizer.w2s[tokens[k]]) >= t[k]

        # CONSTRAINTS FOR FINDING OPTIMAL SOLUTIONS
        for sentence_set in excluded_solutions:
            prob += pulp.lpSum([s[j] for j in sentence_set]) <= len(sentence_set) - 1

        # prob.writeLP('test.lp')

        # solving the ilp problem
        try:
            print('Solving using CPLEX')
            prob.solve(pulp.CPLEX(msg=0))
        except:
            print('Fallback to mentioned solver')
            if solver == 'gurobi':
                prob.solve(pulp.GUROBI(msg=0))
            elif solver == 'glpk':
                prob.solve(pulp.GLPK(msg=0)) 
            else:
                sys.exit('no solver specified')
            
        # retreive the optimal subset of sentences
        solution = set([j for j in range(S) if s[j].varValue == 1])

        # returns the (objective function value, solution) tuple
        return (pulp.value(prob.objective), solution)

    def get_feature_vector(self):
        """
        assign each concept a vector in word2vec space that is the mean of its constituting words

        :return:
        """
        '''
        corpus = [' '.join(doc) for _, doc in docs]
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(corpus)
        idf = vectorizer._tfidf.idf_
        tf_idf = dict(zip(vectorizer.get_feature_names(), idf))
        print tf_idf
        '''
        index = 0
        self.uncertainity, self.concept_vec_idx = {}, {}
        self.fvector = []
        unknown_l, hit_l = [], []
        
        for i in range(len(self.summarizer.sentences)):
            '''
            print(self.summarizer.sentences[i].concepts)
            print(self.summarizer.sentences[i].untokenized_form)
            print(self.summarizer.sentences[i].tokens_pos)
            '''
            # for each concept
            for concept in self.summarizer.sentences[i].concepts:
                #print(self.summarizer.sentences[i].untokenized_form)
                pos_map = [0.0, 0.0, 0.0, 0.0, 0.0] #NN, VB, JJ, ADV, Others 
                if concept not in self.concept_vec_idx:
                    ngram = concept.split(' ')
                    is_capital, is_num, stopword, pos_list, concept_tf, embd = 0, 0, 0, [], [], []
                    for token in ngram:
                        try:
                            word, pos = self.summarizer.sentences[i].tokens_pos[token].split('::')
                        except:
                            token = re.sub(u'[-\.](\s|$)', u'\\1', token)
                            token = re.sub(u'([^.])[.]$', u'\\1', token)
                            try:
                                word, pos = self.summarizer.sentences[i].tokens_pos[token].split('::')
                            except:
                                if token.isnumeric():
                                    word, pos = token, 'CD'
                                else:
                                    word, pos = token, 'NN'
                        if word.istitle():
                            is_capital += 1
                        """
                        if pos == 'CD':
                            is_num += 1
                        if re.match('N.*', pos):
                            pos_map[0] = 1.0
                        if re.match('V.*', pos):
                            pos_map[1] = 1.0
                        if re.match('JJ.*|', pos):
                            pos_map[1] = 1.0
                        """
                        #print(token,)                        
                        if token in self.stoplist:
                            stopword += 1
                        if token in self.summarizer.word_frequencies:
                            concept_tf.append(self.summarizer.word_frequencies[token])
                        if token not in self.stoplist:
                            word_l = word.lower()
                            if word_l in self.embeddings.vocab_dict:
                                embd_val = self.embeddings.W[self.embeddings.vocab_dict[word_l]]
                                hit_l.append(word_l)
                                embd.append(embd_val.tolist())
                            else:
                                joint_words = word_l.split('-')
                                for j_word in joint_words:
                                    j_word = unicode(j_word)
                                    if j_word in self.embeddings.vocab_dict:
                                        embd_val = self.embeddings.W[self.embeddings.vocab_dict[j_word]]
                                        hit_l.append(j_word)
                                        embd.append(embd_val.tolist())
                                    else:
                                        if self.language == "english":
                                            embd_val = self.embeddings.W[self.embeddings.vocab_dict[u"unk"]]
                                        if self.language == "german":
                                            embd_val = self.embeddings.W[self.embeddings.vocab_dict[u"unknown"]]    
                                        unknown_l.append(unicode(word_l))
                                        embd.append(embd_val.tolist())
                                    
                    pos_key = '_'.join(pos_list)
                    if pos_key in self.pos_hash:
                        pos_val = self.pos_hash[pos_key]
                    else:
                        pos_val = len(self.pos_hash) + 1
                        self.pos_hash[pos_key] = pos_val
                    if concept_tf == []:
                        concept_tf = [1]
    
                    # calculate concept vector as the mean of its constituent word vectors.
                    if concept not in self.concept_vec_idx:
                        if not embd:
                            print(embd, concept)
                            if self.language == "english":            
                                embd_val = self.embeddings.W[self.embeddings.vocab_dict[u"unk"]]
                            if self.language == "german":
                                embd_val = self.embeddings.W[self.embeddings.vocab_dict[u"unknown"]]
                            embd.append(embd_val.tolist())
                        vector = np.mean(np.array(embd), axis=0)
                        vector = np.append(vector, np.array([-1]), axis=0)
                        self.fvector.append(vector.tolist())
                        """
                        self.fvector.append([1.0 * self.summarizer.weights[concept]/self.cluster_size,
                                    is_capital,
                                    pos_val,
                                    stopword,
                                    np.mean(np.array(concept_tf)),
                                    is_num])
                        """
                        self.uncertainity[concept] = 1.0
                        self.concept_vec_idx[concept] = index
                        self.index_vec_concept[index] = concept
                        index += 1
        
        hit_l, unknown_l = set(hit_l), set(unknown_l)
        hit, unknown = len(hit_l), len(unknown_l)
        print('size of the feature vector: %d' % len(self.fvector))
        print('hit concepts: %d, unknown concepts: %d' % (hit, unknown))
        print('hit ratio: %f, unknown ratio: %f' % (1.0 * hit/(hit+unknown), 1.0 * unknown/(hit+unknown)))
        print('Unknown words', ','.join(unknown_l))

    def change_labels(self, feedback_list, label):
        for concept in feedback_list:
            # print(concept, self.summarizer.weights[concept])
            vec_index = self.concept_vec_idx[concept]
            self.data[vec_index, -1] = label

    def project_phrase_ngrams(self, concept_list):
        feedback_keys = []
        for phrase in concept_list:
            for key in self.summarizer.weights:
                if re.search(u'(\s|^)%s([\s]|$)' % (key), u'%s' % (phrase)) or re.search(u'(\s|^)%s([\s]|$)' % (phrase),
                                                                                         u'%s' % (key)):
                    # print(key, phrase)
                    feedback_keys.append(key)
        return feedback_keys

    def get_uncertainity_labels(self, model):

        '''
        if self.parse_type == PARSE_TYPE_PARSE:
            #print('Accept keys:', self.recorder.total_accept_keys)
            self.change_labels(self.recorder.total_accept_keys, label=1)
            self.change_labels(self.recorder.union().reject_keys, label=0)
            self.change_labels(self.recorder.union().implicit_reject, label=0)
        if self.parse_type == None:
        '''
        self.change_labels(self.flight_recorder.union().accept, label=1)
        self.change_labels(self.flight_recorder.union().reject, label=0)

        Y = self.data[:, -1]
        X = self.data[:, 1:-1]

        UL_indexes = np.where(Y == -1)
        L_indexes = np.where(Y > -1)

        X_train, Y_train = X[L_indexes], Y[L_indexes]
        X_unlabeled, _ = X[UL_indexes], Y[UL_indexes]

        flag = 0
        try:
            model.fit(X_train, Y_train)
            UL_probs = model.predict_proba(X_unlabeled)
            UL = model.predict(X_unlabeled)
        except:  # If there are no Accepts [training data has only one class]
            flag = 1

        concept_u, concept_labels = {}, {}

        index = 0
        for vec_index in self.index_vec_concept:
            concept = self.index_vec_concept[vec_index]
            if vec_index not in UL_indexes[0]:
                concept_u[concept] = 0.0
                concept_labels[concept] = self.data[vec_index, -1]
            else:
                if flag == 0:
                    prob = UL_probs[index]
                    concept_u[concept] = 1 - prob.max()
                    concept_labels[concept] = UL[index]
                else:  # If there are no Accepts [training data has only one class]
                    concept_u[concept] = 1.0
                    concept_labels[concept] = 1.0
                index += 1

        return concept_u, concept_labels

    def __call__(self, docs, models, summary_length, oracle_type, ub_score, ub_summary, parser_type=None, parse_info=[],
                 max_iteration_count=11, weights_override={}, clear_before_override=None, propagation=False):
        """
        This starts of the simualted feedback for a single cluster of documents, towards a list of models. i.e. the
        models get united, and then the feedback loop is simulated.

        :param docs:
        :param models:
        :param summary_length:
        :param oracle_type:
        :param ub_score:
        :param ub_summary:
        :param parser_type:
        :param parse_info:
        :param max_iteration_count: int: Maximum number of iterations to run.
        :param weights_override: dict: (concept -> double) dictionary containing the override weights for propagation
        """

        self.models = models
        self.summary_length = summary_length
        self.ub_score = ub_score
        self.parse_type = parser_type
        self.cluster_size = len(docs)
        self.MAX_WEIGHT = len(docs)

        for model_name, model in models:
            y = set(extract_ngrams2(model, self.stemmer, self.language, self.N))
            self.ref_ngrams = self.ref_ngrams.union(y)
            if parser_type == PARSE_TYPE_PARSE:
                for _, parse_sents in parse_info[1]:
                    for parse_sent in parse_sents:
                        _, phrases = get_parse_info(parse_sent, self.stemmer, self.language, self.stoplist)
                        y = set(prune_phrases(phrases, self.stoplist, self.stemmer, self.language))
                        self.ref_phrases = self.ref_phrases.union(y)

        self.summarizer.sentences = self.SumeWrap.load_sume_sentences(docs, parser_type, parse_info)
        parse_info = []

        # extract bigrams as concepts
        if self.parse_type == PARSE_TYPE_PARSE:
            print('Get concept types Phrases')
            self.summarizer.extract_ngrams2(concept_type='phrase')
        if self.parse_type == None:
            print('Get concept types ngrams')
            self.summarizer.extract_ngrams2(concept_type='ngrams')

        # compute document frequency as concept weights
        self.summarizer.compute_document_frequency()

        # compute word_frequency
        self.summarizer.compute_word_frequency()
        
        old_sentences = self.summarizer.sentences
        
        self.summarizer.prune_sentences(remove_citations=True, remove_redundancy=True, imp_list=[])

                # from all concepts that are going to be pruned, keep only those that also appear elsewhere

        retained_concepts = [concept for s in self.summarizer.sentences for concept in s.concepts]

        print('Total concepts before sentence pruning: ', len(self.summarizer.weights))
        
        for sentence in set(old_sentences).difference(self.summarizer.sentences):
            for concept in sentence.concepts:
                if concept not in retained_concepts and self.summarizer.weights.has_key(concept):
                    del self.summarizer.weights[concept]

        print('Total concepts found: ', len(self.summarizer.weights))
        
        if self.parse_type == None:
            concept_match = [key for key in self.summarizer.weights if key in self.ref_ngrams]
            print('Total ref concepts:   ', len(self.ref_ngrams))
        elif self.parse_type == PARSE_TYPE_PARSE:
            concept_match = [key for key in self.summarizer.weights if key in self.ref_phrases]
            print('Total ref concepts:   ', len(self.ref_phrases))
        print('UB Accept concepts:   ', len(concept_match))

        if oracle_type.startswith(ORACLE_TYPE_ACTIVE_LEARNING):
            self.get_feature_vector()
            self.data = np.array(self.fvector)
            model = svm.SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced')

        self.initial_weights = self.summarizer.weights

        self.__apply_initial_weights_override__(weights_override, clear_before_override)

        '''
        # create the coocurence graph
        self.graph.clear()
        self.graph.add_sentences(self.summarizer.sentences)
        dump_dir=tempfile.mkdtemp(dir=self.debug_dump_target_dir)
        '''
        
        print('Summarizing %s sentences down to %s words' % (len(self.summarizer.sentences), self.summary_length))
        # core algorithm for feedback calculation... (as in paper)
        flag = 0
                # get_details is the personalizedSummary function which gets updated weights in every iteration.
                # Starting with boudin as starting weights (except in case of weights_override != None).

        # initial iteration
        summary, self.score, subset = self.get_details(1, summary_length, oracle_type)
        self.prev_score = (0.0, 0.0, 0.0)
        prev_summary = ''
        for iteration in range(2, max_iteration_count):
            self.dump_current_weight_map(self.debug_dump_target_dir, max_iteration_count)
            # here, depending on the oracle_type, a intermediate summary is generated. This intermediate summary is
            # satisfies other optimization criteria, so that the amount/probability of getting useful feedback is maximized
            if iteration > 2:
                subset = self.__generate_optimal_feedback_summary__(flag, oracle_type, summary_length)

            print('Summary Subset:', subset)

            # acquire feedback and record it using the flight_recorder
            #new_accepts, new_rejects, new_implicits = self.get_feedback(subset, RECOMMENDER_METHOD_HIGHEST_WEIGHT)
            new_accepts, new_rejects, new_implicits = self.get_feedback(subset)
            self.flight_recorder.record(new_accepts, new_rejects, new_implicits)

            # update the summarizer weights for next iteration
            self.recalculate_weights(oracle_type, propagation)

            summary, self.score, _ = self.get_details(iteration, summary_length, oracle_type)

            if oracle_type.startswith(ORACLE_TYPE_ACTIVE_LEARNING):
                self.uncertainity, self.labels = self.get_uncertainity_labels(model)

            if self.check_break_condition(iteration, prev_summary, summary, ub_summary, self.prev_score):
                break

            self.prev_score = self.score
            prev_summary = summary

        return summary

    def __generate_optimal_feedback_summary__(self, flag, oracle_type, summary_length):
        """
            Generates a summary which is optimal for getting feedback on. This is done by increasing the probability of
            generating a summary with unknown concepts in it. This is achieved by setting the concept weights of known
            concepts (either positivly or negativly rated) to ZERO.

            TODO check if :param subset is neccessary for this method

        :param flag:
        :param oracle_type:
        :param summary_length:
        :return:
        """
        if oracle_type == ORACLE_TYPE_ILP_FEEDBACK or oracle_type.startswith(ORACLE_TYPE_ACTIVE_LEARNING):

            feedback = self.flight_recorder.union().accept | self.flight_recorder.union().reject
            """
            if self.parse_type == PARSE_TYPE_PARSE:
                feedback = self.project_phrase_ngrams(feedback)
            """

            non_feedback = self.summarizer.weights.viewkeys() - feedback
            print("GeOpFeSu: Feedback Size:", len(feedback), len(non_feedback),
                  'Total:', len(self.summarizer.weights.keys()))
            if (self.flight_recorder.latest().accept or len(feedback) == 0) and flag == 0:
                if oracle_type == ORACLE_TYPE_ILP_FEEDBACK:
                    _, subset = self.solve_joint_ilp(int(summary_length), list(feedback), list(non_feedback))
                if oracle_type == ORACLE_TYPE_ACTIVE_LEARNING:
                    _, subset = self.solve_joint_ilp(int(summary_length), list(feedback), list(non_feedback), self.uncertainity)
                if oracle_type == ORACLE_TYPE_ACTIVE_LEARNING2:
                    _, subset = self.solve_joint_ilp(int(summary_length), list(feedback), list(non_feedback), self.uncertainity, self.labels)
                    print('Subset after AL2', subset)
                if not subset:
                    flag = 1
                    print('Solving regular ILP')
                    _, subset = self.summarizer.solve_ilp_problem(summary_size=int(summary_length), units="WORDS")
            else:
                print('Solving regular ILP')
                _, subset = self.summarizer.solve_ilp_problem(summary_size=int(summary_length), units="WORDS")
        else:
            print('Solving regular ILP')
            _, subset = self.summarizer.solve_ilp_problem(summary_size=int(summary_length), units="WORDS")
        return subset

    def __apply_initial_weights_override__(self, weights_override={}, clear_before_override=None):
        """

        :param clear_before_override: bool: if True, all weights are set to a default value, no matter what.
        :param weights_override:
        """
        if (weights_override):
            if clear_before_override is not None:
                print("Clearing summarizer weights")
                for k, v in self.summarizer.weights.iteritems():
                    self.summarizer.weights[k] = float(clear_before_override)
            print("Overriding weights")
            for k, v in weights_override.iteritems():
                if self.summarizer.weights.has_key(k):
                    print("Overriding summarizer weight for '%s' with '%s' (was '%s')" % (
                    k, v, self.summarizer.weights[k]))
                    self.summarizer.weights[k] = v

    def dump_current_weight_map(self, dump_dir=tempfile.mkdtemp(), iteration=0):

        """

        :param dump_dir: directory (has to exist) where the weight map should be stored.
        :param iteration: current iteration
        @type dump_dir: str
        @type iteration: int

        """
        json_content = json.dumps(self.summarizer.weights)
        prefix = "weights-%s-" % iteration
        _, file = tempfile.mkstemp(suffix=".json", prefix=prefix, dir=dump_dir)
        print("Dumping weights to %s" % file)
        write_to_file(json_content, file)
