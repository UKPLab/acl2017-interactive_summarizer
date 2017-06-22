'''
Pipeline code to parse various arguments for Feedback Summarizer

Example:
./pipeline.py --summary_len=100 --oracle_type='reject_all'
'''
from __future__ import print_function, unicode_literals

import datetime
import os.path as path
import sys

from utils.data_helpers import load_w2v_embeddings

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from summarizer.utils.corpus_reader import CorpusReader
from summarizer.baselines.sume_wrap import SumeWrap
from summarizer.algorithms.simulated_feedback import SimulatedFeedback
from summarizer.algorithms.upper_bound_ilp import ExtractiveUpperbound
from summarizer.baselines.sumy.sumy_wrap import sumy_wrap
from summarizer.rouge.rouge import Rouge
from summarizer.utils.writer import write_details_file
from summarizer import settings
import argparse


def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Feedback Summarizer pipeline')
    # -- summary_len: 100, 200, 400
    parser.add_argument('-s', '--summary_size', type=str, help='Summary Length', required=False)

    # --oracle_type:  reject_all, accept_all, weighted, top_n, keep_track
    parser.add_argument('-o', '--oracle_type', type=str, help='Oracle Type: accept_reject, ilp_feedback, active_learning, active_learning2, top_n',
                        required=False, default="")

    # --data_set: DUC2001, DUC2002, DUC2004
    parser.add_argument('-d', '--data_set', type=str, help='Data set ex: DUC2004', required=True)

    # --summarizer_type: upper_bound, feedback
    parser.add_argument('-t', '--summarizer_type', type=str, help='Summarizing Algorithm: upper_bound, sume, feedback',
                        required=True)

    # --parser_type: parse, ner
    parser.add_argument('-p', '--parser_type', type=str, help='Parser info type: parse, ner', required=False)

    # --language: english, german
    parser.add_argument('-l', '--language', type=str, help='Language: english, german', required=True)

    # --rouge: "~"
    parser.add_argument('-r', '--rouge', type=str, help='Rouge: ROUGE directory', required=False,
                        default=settings.ROUGE_DIR)

    # --rouge: "~"
    parser.add_argument('-io', '--iobasedir', type=str, help='IO base directory', required=False,
                        default=path.join(path.dirname(path.abspath(__file__)), "data"))

    args = parser.parse_args()
    return args.summary_size, args.oracle_type, args.data_set, args.summarizer_type, args.parser_type, args.language, args.rouge, args.iobasedir


def get_output_dir(oracle_type, summary_size, data_set, parser_type, iobasedir=""):
    if summary_size != None:
        if parser_type == None:
            woutput = "%s_%s_%s" % (oracle_type, data_set, summary_size)
        else:
            woutput = "%s_%s_%s_%s" % (oracle_type, parser_type, data_set, summary_size)
    else:
        if parser_type == None:
            woutput = "%s_%s" % (oracle_type, data_set)
        else:
            woutput = "%s_%s_%s" % (oracle_type, parser_type, data_set)
    return woutput


def pipeline(summary_size, data_set, summarizer_type, parser_type, language, rouge_dir, iobasedir, oracle_type=""):
    #datasets_path = path.normpath(path.join(iobasedir, "datasets"))
    embeddings_path = path.normpath(path.join(iobasedir, "embeddings"))
    datasets_processed_path = path.join(iobasedir, "processed")
    if not path.isdir(datasets_processed_path):
        print(datasets_processed_path)
        raise "Processed datasets directory doesnt exist. please pre-process data."

    woutput = get_output_dir(oracle_type, summary_size, data_set, parser_type, iobasedir)
    print("Output base dir for writing data is %s" % woutput)

    # data_path = "%s/data" % (path.dirname(path.abspath(__file__)))
    # summary_size, oracle_type, data_set, summarizer_type, parser_type, LANGUAGE, rouge_dir = get_args()
    #
    # woutput = get_output_dir(oracle_type, summary_size, data_set, parser_type)
    print(woutput)
    print("rouge_dir is %s" % rouge_dir)
    rouge = Rouge(rouge_dir)

    summary_len = summary_size

    reader = CorpusReader(datasets_processed_path)
    data = reader.get_data(data_set, summary_len)

    reader2 = CorpusReader(datasets_processed_path, "parse")
    parse_data = reader2.get_data(data_set, summary_len)

    embeddings = load_w2v_embeddings(embeddings_path, language, oracle_type)

    parse_info = []
    for topic, docs, models in data:
        if parser_type or oracle_type.startswith('active_learning'):
            # Parse sentences using stanford parser
            _, parse_docs, parse_models = parse_data.next()

        info_all = []
        filename = path.join(iobasedir, "scores2", data_set, woutput, topic + '.csv')
        print("Topic:%s" % topic)
        if path.isfile(filename):
            continue
        if summarizer_type == 'baselines':
            algos = ['Luhn', 'LexRank', 'TextRank', 'Lsa', 'Kl']
            if summary_size != None:
                summaries = []
                for algo_type in algos:
                    summaries.append(sumy_wrap(docs, algo_type, language, int(summary_len)))

            for model in models:
                print("###")
                if summary_size == None:
                    summaries = []
                    summary_len = len(' '.join(model[1]).split(' '))
                    print('Summary_len: %s' % summary_len)
                    for algo_type in algos:
                        summaries.append(sumy_wrap(docs, algo_type, language, int(summary_len)))

                summarizer = ExtractiveUpperbound(language)
                summary1 = summarizer(docs, [model], summary_len, 1)
                summary2 = summarizer(docs, [model], summary_len, 2)
                R1_1, R2_1, R4_1 = rouge(' '.join(summary1), [model], summary_len)
                R1_2, R2_2, R4_2 = rouge(' '.join(summary2), [model], summary_len)
                model_name = model[0][model[0].rfind('/') + 1:]
                print("Model: %s" % model_name)
                print("UB1 %f %f %f\nUB2: %f %f %f" % (R1_1, R2_1, R4_1, R1_2, R2_2, R4_2))
                for i, summary in enumerate(summaries):
                    R1, R2, R4 = rouge(' '.join(summary), [model], summary_len)
                    print("%s %f %f %f" % (algos[i], R1, R2, R4))
                print("###")
        if summarizer_type == 'upper_bound':
            # not used
            summarizer = ExtractiveUpperbound(language)
            summary = summarizer(docs, models, summary_len)
            R1, R2 = rouge(' '.join(summary), models, summary_len)
            print('%s,%4f,%4f,\"%s\"' % (topic, R1, R2, ' '.join(summary)))
        if summarizer_type == 'sume':
            # actual baseline of Boudin
            summarizer = SumeWrap(language)
            summary = summarizer(docs, summary_len)

        if summarizer_type == 'feedback':
            for m_idx, model in enumerate(models):
                if summary_size == None:
                    summary_len = len(' '.join(model[1]).split(' '))
                print('Summary_len: %s' % summary_len)
                summarizer = ExtractiveUpperbound(language)
                ub_summary = summarizer(docs, [model], summary_len, ngram_type=2)

                summary_txt = '\n'.join(ub_summary)
                ub_score = rouge(summary_txt, [model], summary_len)
                info = []

                info.append(['0', ub_score[0], ub_score[1], ub_score[2], 0, 0, summary_txt])
                # prints the 3 different rouge scores that have been calculated.

                print(info[0][:-1])
                # prints info array except last item (summary_txt)

                summarizer = SimulatedFeedback(language, rouge, embeddings)

                if parser_type or oracle_type.startswith('active_learning'):
                    parse_info = [parse_docs, [parse_models[m_idx]]]

                summarizer(docs, [model], summary_len, oracle_type, ub_score, ub_summary, parser_type, parse_info,
                           max_iteration_count=11, propagation=False)

                info.extend(summarizer.info_data)
                info_all.append(info)

            write_details_file(info_all, path.join(iobasedir, "scores2", data_set, woutput, topic + '.csv'))


if __name__ == '__main__':
    summary_size, oracle_type, data_set, summarizer_type, parser_type, lang, rouge_dir, iobasedir = get_args()

    print("+---------------- Running the pipeline ------------------")
    print("| configuration settings:")
    print("| summary_size", summary_size)
    print("| data_set", data_set)
    print("| oracle_type", oracle_type)
    print("| summarizer_type", summarizer_type)
    print("| parser_type", parser_type)
    print("| lang ", lang)
    print("| rouge_dir ", rouge_dir)
    print("| iobasedir", iobasedir)
    print("+--------------------------------------------------------")

    start = datetime.datetime.now()
    print("starting on ",  str(start))
    print("+--------------------------------------------------------")

    pipeline(summary_size, data_set, summarizer_type, parser_type, lang, rouge_dir, path.expanduser(iobasedir), oracle_type)
    print("+--------------------------------------------------------")
    end = datetime.datetime.now()
    print("| stopped on ",  str(end))
    print("| total time: ", str(end - start))
    print("+--------------------------------------------------------")


