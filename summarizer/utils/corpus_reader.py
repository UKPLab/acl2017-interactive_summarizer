import sys
import os.path as path
import codecs

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
import re, os


class CorpusReader(object):
    def __init__(self, base_path, parse_type=None):
        self.base_path = base_path
        self.parse_type=parse_type

    def load_processed(self, path, summary_len=None):
        data = []
        docs = os.listdir(path)
        if summary_len:
            summaries = [model for model in docs if re.search("M\.%s\." % (summary_len), model)]
            docs = summaries
        for doc_name in docs:
            filename = "%s/%s" % (path, doc_name)
            with codecs.open(filename, 'r', 'utf-8') as fp:
                text = fp.read().splitlines()
            data.append((filename, text))
        return data

    def get_data(self, corpus_name, summary_len):
        """
        generator function that returns a iterable tuple which contains

        :rtype: tuple consisting of topic, contained documents, and contained summaries
        :param corpus_name: 
        :param summary_len:
        """
        corpus_base_dir = path.join(self.base_path, corpus_name)

        docs_directory_name = "docs"
        models_directory_name = "summaries"
        if self.parse_type == "parse":
            docs_directory_name = "docs.parsed"
            models_directory_name = "summaries.parsed"

        dir_listing = os.listdir(corpus_base_dir)
        for ctopic in dir_listing:
            docs_path = path.join(corpus_base_dir, ctopic, docs_directory_name)
            summary_path = path.join(corpus_base_dir, ctopic, models_directory_name)

            docs = self.load_processed(docs_path)
            summaries = self.load_processed(summary_path, summary_len)
            yield ctopic, docs, summaries
