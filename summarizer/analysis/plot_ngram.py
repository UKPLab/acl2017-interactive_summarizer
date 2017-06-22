'''
Code to print plots of n-gram match

Example:
./pipeline.py --summary_len=100 --oracle_type='reject_all'
'''
import sys, os.path as path

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from summarizer.utils.corpus_reader import CorpusReader
from summarizer.baselines.sume_wrap import SumeWrap
from summarizer.algorithms.upper_bound_ilp import ExtractiveUpperbound
from summarizer.rouge.rouge import Rouge
from summarizer.settings import ROUGE_DIR
from sets import Set
import argparse
import matplotlib.pyplot as plt
from summarizer.utils.data_helpers import get_sorted, extract_ngrams2
from nltk.stem.snowball import SnowballStemmer

def ngrams_match(ref_ngrams, summ_ngrams):
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
    return list(Set(ref_ngrams) & Set(summ_ngrams))


def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Feedback Summarizer pipeline')
    # -- summary_len: 100, 200, 400
    parser.add_argument('-s', '--summary_size', type=str, help='Summary Length ex:100', required=True)

    # --data_set: DUC2001, DUC2002, DUC2004
    parser.add_argument('-d', '--data_set', type= str, help='Data set ex: DUC2004', required=True)
    
    parser.add_argument('-l', '--language', type= str, help='Language ex: english, german', required=True)

    args = parser.parse_args()
    summary_len = args.summary_size
    data_set = args.data_set
    language = args.language
    return summary_len, data_set, language

def plot_ngrams():
    data_path = "%s/data" % (path.dirname(path.dirname(path.abspath(__file__))))
    summary_len, data_set, language = get_args()
    stemmer = SnowballStemmer(language) 

    reader = CorpusReader(data_path)
    data = reader.get_data(data_set, summary_len)
    
    for topic, docs, models in data:
        
        print topic
        summarizer = ExtractiveUpperbound(language)
        ub_summary = summarizer(docs, models, summary_len, ngram_type=2)
            
        summarizer = SumeWrap(language)
        summarizer.s.sentences = summarizer.load_sume_sentences(docs)
        summarizer.s.extract_ngrams2()
        summarizer.s.compute_document_frequency()
        
        sorted_list = get_sorted(summarizer.s.weights)
        
        ngrams_ub = extract_ngrams2(ub_summary, stemmer, language)
        ngrams_models = []
        for _, model in models:
            ngrams_models.append(extract_ngrams2(model, stemmer, language))
        
        inter_ngrams = []
        for i in range(len(ngrams_models)):
            for j in range(i+1, len(ngrams_models)):
                inter_ngrams.extend(Set(ngrams_models[i]).intersection(Set(ngrams_models[j])))
        
        final_ngrams_models = []
        for i in range(len(ngrams_models)):
            final_ngrams_models.append(list(Set(ngrams_models[i])-Set(inter_ngrams)))                

        all_ngrams_unique = []
        for ngrams in ngrams_models:
            all_ngrams_unique = list(Set(all_ngrams_unique).union(Set(ngrams)))
        
        all_ngrams = []
        for ngrams in ngrams_models:
            all_ngrams.extend(ngrams)
        all_ngrams = list(Set(all_ngrams))
        
        final_ngrams_models.append(all_ngrams)
    
        x = [0]
        y = [[0] for _ in range(len(final_ngrams_models))]
        
        for i in range(50, len(sorted_list), 10):
            docs_ngrams = sorted_list[:i]
            x.append(i)
            prev_y = 0
            for index in range(len(final_ngrams_models)):
                val = len(ngrams_match(docs_ngrams, final_ngrams_models[index]))
                if index == len(final_ngrams_models)-1:
                    y[index].append(val)
                else:
                    y[index].append(prev_y + val)
                prev_y += val
        plt.fill_between(x, [0]*len(y[0]), y[0], facecolor='green',interpolate=True)
        plt.plot(x, y[0], 'g', label='Unique bigrams by User 1')
        plt.fill_between(x, y[0], y[1], facecolor='blue',interpolate=True)
        plt.plot(x, y[1], 'r', label='Unique bigrams by User 2')
        plt.fill_between(x, y[1], y[2], facecolor='red',interpolate=True)
        plt.plot(x, y[2], 'b', label='Unique bigrams by User 3')
        plt.fill_between(x, y[2], y[3], facecolor='yellow',interpolate=True)
        plt.plot(x, y[3], 'y', label='Unique bigrams by User 4')
        plt.fill_between(x, y[3], y[4], facecolor='black',interpolate=True)
        plt.plot(x, y[4], 'k', label='Overlapping bigrams between atleast two Users')
        
        #plt.plot(x, y[5], 'k', label='Upper Bound')
        
        plt.legend(loc="upper left", fontsize=10)
        plt.xlabel("No. of sorted bigrams in the source documents", fontsize=20)
        plt.ylabel("Overlapping w.r.t. reference summaries", fontsize=20)
        plt.yscale("linear", linewidth=1)
        plt.grid(True)
        plt.show()
        
if __name__ == '__main__':
    plot_ngrams()
