import sys, os.path as path
import os
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import argparse

from corpus_cleaner import CorpusCleaner

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)
    #DUC2001, DUC2002, DUC2004
    parser.add_argument('-d', '--data_set', help="Data set to process. Basically a directory name.", type=str, required=True)

    #../data
    parser.add_argument('-p', '--data_path', help="Data Path, Defaults to '$PWD/data'", type=str, required=False,
                        default=path.join(os.getcwd(), "data"))

    #parse, NER
    parser.add_argument('-a', '--annotation_type', help="Annotation Type", type=str, required=False)


    parser.add_argument('-l', '--language', help="Language", type=str, required=False)

    # iobasedir
    # parser.add_argument('-i', '--iobase',
    #                     help="base directory",
    #                     type=str,
    #                     required=False,
    #                     default=path.normpath(path.join(path.expanduser("~"), ".ukpsummarizer")))
    args = parser.parse_args()

    corpus_name = args.data_set
    parse_type = args.annotation_type
    language = args.language
    # iobase_dir = args.iobase
    data_path = path.normpath(args.data_path)

    if parse_type !=None and language==None:
        raise AttributeError('Please specify language')

    corpus = CorpusCleaner(data_path, corpus_name, parse_type, language)
    if corpus_name[:3] == 'DUC' or corpus_name[:3] == 'TAC':
        corpus.cleanDuc_data(parse_type)
    if corpus_name[:3] == 'DBS':
        corpus.cleanDBS_data(parse_type)
    if corpus_name == 'WikiAIPHES':
        corpus.cleanWiki_data(parse_type)
    else:
        pass

if __name__ == '__main__':
    main()
