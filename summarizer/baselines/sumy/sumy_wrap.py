from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import os.path as path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from summarizer.baselines.sumy.sumy.parsers.plaintext import PlaintextParser
from summarizer.baselines.sumy.sumy.nlp.tokenizers import Tokenizer
from summarizer.baselines.sumy.sumy.summarizers.lsa import LsaSummarizer
from summarizer.baselines.sumy.sumy.summarizers.kl import KLSummarizer
from summarizer.baselines.sumy.sumy.summarizers.luhn import LuhnSummarizer
from summarizer.baselines.sumy.sumy.summarizers.lex_rank import LexRankSummarizer
from summarizer.baselines.sumy.sumy.summarizers.text_rank import TextRankSummarizer
from summarizer.baselines.sumy.sumy.nlp.stemmers import Stemmer
from nltk.corpus import stopwords

def sumy_wrap(documents, summarizer_type, LANGUAGE, SUMMARY_SIZE):
    doc_string = u'\n'.join([u'\n'.join(sents) for doc, sents in documents]) 
    
    parser = PlaintextParser.from_string(doc_string, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    if summarizer_type=='Lsa':
        summarizer = LsaSummarizer(stemmer)
    if summarizer_type=='Kl':
        summarizer = KLSummarizer(stemmer)
    if summarizer_type=='Luhn':
        summarizer = LuhnSummarizer(stemmer)
    if summarizer_type=='LexRank':
        summarizer = LexRankSummarizer(stemmer)
    if summarizer_type=='TextRank':
        summarizer = TextRankSummarizer(stemmer)
    
    summarizer.stop_words = frozenset(stopwords.words(LANGUAGE))
    summary = summarizer(parser.document, SUMMARY_SIZE)
    return summary 