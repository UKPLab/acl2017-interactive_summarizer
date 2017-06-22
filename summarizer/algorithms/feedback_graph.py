from __future__ import print_function

import tempfile
from abc import abstractmethod, ABCMeta
from collections import Counter

import datetime
import time
import networkx as nx
from nltk import ngrams

from algorithms.base import Sentence
from algorithms.flight_recorder import Record, FlightRecorder
from utils.data_helpers import sent2stokens
from operator import itemgetter



class FeedbackStore(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def incorporate_feedback(self, flightrecorder):
        """
            incorporates the feedback object into the graph, incl. propagation among neighbors
        :param flightrecorder: FlightRecorder
        @type flightrecorder: FlightRecorder
        """
        pass

    @abstractmethod
    def get_weights(self):
        """
            returns a generator which provides "concept -> weight" tuples, and weight should be a float in the range
            of [0..1]. i.e. the resulting weights that can be plumbed into the ILP-Solver
        """
        pass

    @abstractmethod
    def clear(self):
        """
        cleans this instance, so it can be reused.

        :return: None
        """
        pass


class SimpleNgramFeedbackGraph(FeedbackStore):
    def __init__(self, stemmer, language, N=2, G=nx.Graph()):
        """
        This class does some simple propagation based feedback on a token-graph. Nodes represent Tokens (stemmed), and
        edges represent N-Grams. As default, :param N is "2", and the Graph is directed.
        """

        self.G = G
        """ the graph which has encoded the concept weight along its edges. """

        self.counter = Counter()
        self.language = language
        self.stemmer = stemmer
        self.N = N

    def add_sentences(self, sentences):
        """
        @type sentences: list[Sentence]
        """
        counter = self.counter
        G = self.G
        for sent in sentences:
            counter.update(ngrams(sent.tokens, self.N))
            G.add_nodes_from(sent.tokens)

        updated_edges = []
        for v in counter.elements():
            s = v[0]
            t = v[1]
            c = counter[v]
            updated_edges.append((s, t, c))

        G.add_weighted_edges_from(updated_edges)

    def incorporate_feedback(self, flightrecorder):
        # print("Incorporating feedback into graph")
        G = self.G
        maxweight = self.counter.most_common()[0][1]
        latest_feedback = flightrecorder.latest()

        for edge_as_string in latest_feedback.accept:
            splitted = edge_as_string.split(" ")

            u = splitted[0]
            v = splitted[1]
            if not G.has_edge(u, v):
                continue

            edge = G.get_edge_data(u, v)
            weight = edge.get("weight")

            # set the edge itself to max weight
            G.add_edge(u, v, {"weight": maxweight})

            # multiply adjacent edges by 2
            for n in G.successors(u):
                if n is v:
                    continue
                w = G.get_edge_data(u, n).get("weight")
                newweight = min(maxweight, w * 2.0)
                G.add_edge(u, n, {"weight": newweight})

            for n in G.successors(v):
                if n is u:
                    continue
                w = G.get_edge_data(v, n).get("weight")
                newweight = min(maxweight, w * 2.0)
                G.add_edge(v, n, {"weight": newweight})

        for edge_as_string in latest_feedback.reject:
            splitted = edge_as_string.split(" ")

            u = splitted[0]
            v = splitted[1]
            if not G.has_edge(u, v):
                continue

            edge = G.get_edge_data(u, v)
            weight = edge.get("weight")

            # set the edge itself to max weight
            G.add_edge(u, v, {"weight": 0.0})

            # multiply adjacent edges by 2
            for n in G.successors(u):
                if n is v:
                    continue
                w = G.get_edge_data(u, n).get("weight")
                newweight = max(0, w / 1.2)
                G.add_edge(u, n, {"weight": newweight})

            for n in G.successors(v):
                if n is u:
                    continue
                w = G.get_edge_data(v, n).get("weight")
                newweight = max(0, w / 1.2)
                G.add_edge(v, n, {"weight": newweight})

    def get_weights(self):
        G = self.G
        maxweight = self.counter.most_common()[0][1]
        # get the largest count to scale weights between 0 and 1.
        for (u, v, attr) in G.edges_iter(data="weight"):
            weight = attr
            ngram = u + " " + v
            yield (ngram, float(weight / maxweight))

    def clear(self):
        self.G.clear()
        self.counter.clear()


class PageRankFeedbackGraph(FeedbackStore):
    """
    PageRankFeedbackGraph uses the pagerank algorithm to infer node weights, which act as
    """

    def __init__(self, stemmer, language, N=2, G=nx.DiGraph()):
        self.G = G
        self.stemmer = stemmer
        self.language = language
        self.N = N

        self.counter = Counter()
        self.pr = nx.pagerank(G)

    def add_sentences(self, sentences):
        """
        @type sentences: list[Sentence]
        :param sentences:
        :return:
        """
        counter = self.counter
        G = self.G
        for sentence in sentences:
            G.add_nodes_from(sentence.concepts)
            counter.update(ngrams(sentence.concepts, self.N))

        for (keys, value) in counter.items():
            for i in range(0, len(keys) - 1):
                for j in range(1, len(keys)):
                    G.add_edge(keys[i], keys[j], weight=value)
                    # counter.update((keys[i], keys[j]))

        # for (key, value) in counter.items():
        #     G.add_edge(key[0], key[1], attr={"weight": value})

        print("V := (N,E), |N| = %s, |E| = %s" % (len(G.nodes()), len(G.edges())))

        self.pr = nx.pagerank(G)

    def get_weights(self):
        G = self.G
        pr = self.pr
        max_pagerank = max(pr.itervalues())
        # get the largest count to scale weights between 0 and 1.

        t = datetime.datetime.now()
        ts = int(time.mktime(t.timetuple()))
        temp = tempfile.mktemp(prefix=str(ts), suffix=".gexf")

        nx.write_gexf(G, temp)

        for (k, v) in pr.iteritems():
            yield (k, float(v / max_pagerank))

    def clear(self):
        pass

    def incorporate_feedback(self, flightrecorder):
        """

        :param flightrecorder:
        :return:
         @type flightrecorder: FlightRecorder
        """
        G = self.G
        print("V := (N,E), |N| = %s, |E| = %s" % (len(G.nodes()), len(G.edges())))

        # use the pagerank personalization feature to incorporate flightrecorder feedback

        union = flightrecorder.union()

        for rejected in union.reject:
            if(G.has_node(rejected)):
                G.remove_node(rejected)

        print("V := (N,E), |N| = %s, |E| = %s" % (len(G.nodes()), len(G.edges())))

        self.pr = nx.pagerank(G)
