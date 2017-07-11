import sys, os.path as path
import codecs
import unicodedata
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from pyclausie import ClausIE

from summarizer.utils.reader import read_file
from shutil import copyfile
import os, re
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
from summarizer.data_processer.task_extractor import TaskExtractor
from nltk.parse import stanford

import string
from summarizer.utils.data_helpers import text_normalization, remove_spaces_lines
from summarizer.utils.writer import write_to_file, create_dir, clean_create_dir
from summarizer.data_processer.clauseIE_tree import create_trees
from nltk.tokenize import word_tokenize

PROJECT_PATH = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
PUNCT = tuple(string.punctuation + "'")


class CorpusCleaner(object):
    def __init__(self, datasets_path, corpus_name, parse_type, lang='english'):
        self.datasets_path = datasets_path
        self.corpus_name = corpus_name
        self.corpus_path = path.join(datasets_path, corpus_name)
        self.docs_path = path.join(self.corpus_path, "docs")
        self.topics_file = path.join(self.corpus_path, "topics.xml")
        self.models_path = path.join(self.corpus_path, "models")
        self.smodels_path = path.join(self.corpus_path, "smodels")
        self.jar_path = path.join(PROJECT_PATH, "summarizer", "jars")
        os.environ['CLASSPATH'] = self.jar_path
        self.cleaned_path = path.join(datasets_path, "processed")

        if parse_type == 'parse':
            if lang == 'english':
                self.parser = stanford.StanfordParser(model_path="%s/englishPCFG.ser.gz" % (self.jar_path))
            if lang == 'german':
                self.parser = stanford.StanfordParser(model_path="%s/germanPCFG.ser.gz" % (self.jar_path))
                # self.cleaned_path = path.join(datasets_path, "processed.parse")
        if parse_type == 'props':  # TODO
            if lang == 'english':
                self.props_parser = ClausIE.get_instance()
            if lang == 'german':
                self.parser = stanford.StanfordParser(model_path="%s/germanPCFG.ser.gz" % (self.jar_path))
            

    def parse_xml_all(self, data_file, doc_type, language='english'):
        e = ET.parse(data_file)
        cluster_data = {}
        root = e.getroot()
        for topics in root:
            data = []
            topic_id = topics.attrib.get('id')
            for documents in topics.findall(doc_type):
                doc_id = documents.attrib.get('id')
                if doc_type == 'document':
                    title_text = documents.find('title').text
                doc_text = documents.find('text').text
                text = text_normalization(doc_text)
                doc_sents = sent_tokenize(text, language)
                data.append([doc_id, doc_sents])
            cluster_data[topic_id] = data
        return cluster_data

    def load_processDBS(self, docs_file, summaries_file):
        docs_data = self.parse_xml_all(docs_file, 'document', language='german')
        summaries_data = self.parse_xml_all(summaries_file, 'summary', language='german')
        return docs_data, summaries_data

    def loadWiki_cluster(self, ctopic):
        cluster_path = '%s/%s/' % (self.corpus_path, ctopic)
        self.docs_path = '%s/input/' % (cluster_path)
        self.models_path = '%s/reference/' % (cluster_path)
        docs, summaries = [], []
        for datafile in os.listdir(self.docs_path):
            if datafile.startswith('M'):
                filename = '%s/%s' % (self.docs_path, datafile)

                with codecs.open(filename, 'r', 'utf-8', errors='ignore') as fp:
                    data = fp.read().splitlines()
                text = ' '.join(data)
                #print text.encode('utf-8')
                text = re.sub(u'\u201e|\u201c', u'', text)
                text = re.sub(u'\(EL[^\)]+\)', u'', text)
                text = re.sub(u'\(MLI[^\)]+\)', u'', text)
                text = re.sub(u'\(ED[^\)]+\)', u'', text)
                text = re.sub(u'[0-9]+([A-Z])', u'. \\1', text)
                text = re.sub(u'^\. ', u'', text)
                text = re.sub(u'\u2026', u'.', text)
                text = re.sub(u'\u00E2', u'', text)
                text = re.sub(u'\u00E0', u'', text)
                text = re.sub(u'; \*', u'.', text)
                text = re.sub(u'([0-9]\))\n', u'\\1.\n', text)
                text = text_normalization(text)
                new_data = sent_tokenize(text)
                docs.append((datafile, new_data))
        for model_file in os.listdir(self.models_path):
            filename = '%s/%s' % (self.models_path, model_file)
            with codecs.open(filename, 'r', 'utf-8') as fp:
                data = fp.read().splitlines()
            summaries.append((model_file, data))
        return (docs, summaries)

    def cleanWiki_data(self, parse_type):
        for ctopic in os.listdir(self.datasets_path):
            docs, summaries = self.loadWiki_cluster(ctopic)
            print 'TOPIC:', ctopic
            if parse_type == 'parse':
                parsed_docs = self.runparser_data(docs)
                parsed_summaries = self.runparser_data(summaries)
                self.create_processed(ctopic, parsed_docs, doc_type='docs.parsed')
                self.create_processed(ctopic, parsed_summaries, doc_type='summaries.parsed')
            self.create_processed(ctopic, docs, doc_type='docs')
            self.create_processed(ctopic, summaries, doc_type='summaries')

    def cleanDBS_data(self, parse_type):
        docs_file = "%s/dbs-documents.xml" % (self.docs_path)
        summaries_file = "%s/dbs-summary.xml" % (self.models_path)
        docs_data, summaries_data = self.load_processDBS(docs_file, summaries_file)
        for ctopic in docs_data:
            docs, summaries = docs_data[ctopic], summaries_data[ctopic]
            dir_path = path.join(self.cleaned_path, self.corpus_name, ctopic, "docs")
            if parse_type == 'parse':
                parsed_docs = self.runparser_data(docs)
                parsed_summaries = self.runparser_data(summaries)
                self.create_processed(ctopic, parsed_docs, doc_type='docs.parsed')
                self.create_processed(ctopic, parsed_summaries, doc_type='summaries.parsed')
            self.create_processed(ctopic, docs, doc_type='docs')
            self.create_processed(ctopic, summaries, doc_type='summaries')

    def create_processed(self, ctopic, docs, doc_type):
        for doc_name, doc_sents in docs:
            data = '\n'.join(doc_sents)
            dir_path = path.join(self.cleaned_path, self.corpus_name, ctopic, doc_type)

            create_dir(dir_path)
            filename = path.join(dir_path, doc_name)
            write_to_file(data, filename)

    def runparser_data(self, docs):
        new_docs = []
        for doc_name, doc in docs:
            print 'Processing:', doc_name
            sentences = self.parser.raw_parse_sents(doc)
            sents = []
            for sent in sentences:
                parsestr = unicode(list(sent)[0])
                sents.append(remove_spaces_lines(parsestr))
            new_docs.append((doc_name, sents))
        return new_docs
    
    def props_exception(self, doc_name, doc):
        if doc_name=='NYT19981114.0057':
            doc[23]= doc[23].replace('but who belong to no group,', '')
        if doc_name=='APW19981005.0474':
            doc[6]= doc[6].replace('what it claims are ', '')
        return doc
    
    def runprops_data(self, docs):
        new_docs = []
        for doc_name, doc in docs:
            print 'Processing:', doc_name
            doc_new = []
            doc = self.props_exception(doc_name, doc)
            
            for index, sent in enumerate(doc):
                doc_new.append(' '.join(word_tokenize(sent)))
                print index+1, doc_new[index]
            
            triples = []
            for i, sent in enumerate(doc_new):
                try:
                    tmp_triples = self.props_parser.extract_triples([sent])
                    triples.append(tmp_triples)
                except:
                    print('Error: failed for line %s' % (sent))
                    continue
            parse_sents = create_trees(triples, doc_new)
            sents = []
            new_docs.append((doc_name, parse_sents))
        return new_docs

    def cleanDuc_data(self, parse_type):
        if path.exists(self.topics_file):
            task_extractor = TaskExtractor(path.join(self.cleaned_path, self.corpus_name))
            task_extractor.process(self.topics_file)
        for ctopic in os.listdir(self.docs_path):
            print "Cleaning ", ctopic
            try:
                docs, summaries = self.clean_DUC_cluster(ctopic)
                dir_path = path.join(self.cleaned_path, self.corpus_name, ctopic, "docs.props")
                
                if path.isdir(dir_path):
                    print 'Cleaning:', dir_path
                    #clean_create_dir(dir_path)
                    continue
                
                if parse_type == 'parse':
                    parsed_docs = self.runparser_data(docs)
                    parsed_summaries = self.runparser_data(summaries)
                    self.create_processed(ctopic, parsed_docs, doc_type='docs.parsed')
                    self.create_processed(ctopic, parsed_summaries, doc_type='summaries.parsed')
                if parse_type == 'props':
                    parsed_docs = self.runprops_data(docs)
                    parsed_summaries = self.runprops_data(summaries)
                    self.create_processed(ctopic, parsed_docs, doc_type='docs.props')
                    self.create_processed(ctopic, parsed_summaries, doc_type='summaries.props')
                    
                self.create_processed(ctopic, docs, doc_type='docs')
                self.create_processed(ctopic, summaries, doc_type='summaries')
            except 'Exception':
                t, v, st = sys.exc_info()
                print "error cleaning %s - type: %s, value %s" % (ctopic, t, v)
                exit(0)

    def copy_ssummaries(self, topic_summ, smodels_org, smodels_dir):
        for ssumm_file in topic_summ:
            rfile = '/'.join([smodels_org, ssumm_file])
            wfile = '/'.join([smodels_dir, ssumm_file])
            copyfile(rfile, wfile)

    def extractSummaries(self, ctopic):
        cid = "%s.M." % (ctopic)
        model_files = [filename for filename in os.listdir(self.models_path) if filename.startswith(cid)]
        models = []
        for model_file in model_files:
            model_name = path.join(self.models_path, model_file)
            fp = open(model_name, 'r')
            text = fp.read()
            model_text = sent_tokenize(' '.join(text.replace('\n', ' ').split()))
            models.append((model_file, model_text))
        return models

    def clean_DUC_cluster(self, ctopic):
        cluster_id = ctopic[:-1].upper()
        summaries = self.extractSummaries(cluster_id)
        docs = []
        cluster_path = path.join(self.docs_path, ctopic)
        for datafile in os.listdir(cluster_path):
            filename = path.join(cluster_path, datafile)
            file_data = read_file(filename).replace('&', '111')
            print 'Going to parse xml: '"%s"' ; %s ' % (filename, type(file_data))
            doc_no, headline, rawtext = self.parse_xml(file_data, filename)
            docs.append((doc_no, rawtext))
        return (docs, summaries)

    def parse_xml(self, file_data, filename="unknown filename"):
        e = ET.fromstring(file_data)
        if self.corpus_name == "DUC2002" or self.corpus_name == "DUC2001" or self.corpus_name == "DUC2006" or self.corpus_name == "DUC2007":
            text = ""
            if self.corpus_name == 'DUC2006' or self.corpus_name == 'DUC2007':
                textid = e.find('BODY').findall('TEXT')
                headid = e.find('HEADLINE')
            else:
                textid = e.findall('TEXT')
                headid = e.find('HEAD')
            for text_block in textid:
                paras = text_block.findall('P')
                if not paras:
                    text += text_block.text
                elif paras:
                    text = ""
                    for para in paras:
                        tmp_txt = para.text.rstrip(' \n')
                        if re.search(r'[\'".?]$', tmp_txt):
                            text += tmp_txt + ' '
                        else:
                            text += tmp_txt + '. '
            if headid == None:
                headid = e.find('HL')
                if headid == None:
                    headid = e.find('HEADLINE')
                    if headid == None:
                        headline = ""
                    else:
                        paras = e.find('HEADLINE').findall('P')
                        if paras == None:
                            headline = headid.text.replace('\n', ' ')
                        else:
                            headline = ""
                            for para in paras:
                                headline += para.text.replace('\n', ' ')
                else:
                    headline = headid.text.replace('\n', ' ')
            elif headid != None:
                headline = headid.text.replace('\n', ' ')

            text = re.sub(u"(ATHLETE:)", u". \\1", text)
            text = re.sub(u"[*]\n", u". ", text)
        elif self.corpus_name == "DUC2004" or self.corpus_name == "DUC_TEST":
            text = e.find('TEXT').text
            headline = ""
        elif self.corpus_name == "DUC2004TASK5":
            text = ""
            textids = e.find('BODY').findall('TEXT')
            for id in textids:
                paras = id.findall('P')
                if not paras:
                    text += id.text
                elif paras:
                    for para in paras:
                        tmp_txt = para.text.rstrip(' \n')
                        if re.search(r'[\'".?]$', tmp_txt):
                            text += tmp_txt + ' '
                        else:
                            text += tmp_txt + '. '
            headline = e.find("HEADLINE")
            if not headline:
                headline = ""

        # TODO add DUC2007 parsing
        print "Original_text:", text.encode('utf-8')
        text = text.replace(u"111amp;",u"")
        text = re.sub(u'111[^;]+;', u'', text)
        text = text_normalization(text)
        raw_sents = sent_tokenize(text)

        doc_no = e.find('DOCNO').text.replace(' ', '')
        doc_no = doc_no.replace('\n', '')
        return doc_no, headline, raw_sents

# def ensure_unicode(v):
#     if isinstance(v, str):
#         v = v.decode('utf-8')
#     return unicode(v)  # convert anything not a string to unicode too
