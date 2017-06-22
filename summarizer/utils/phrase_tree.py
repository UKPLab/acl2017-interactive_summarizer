# -*- coding: utf-8 -*-
import os
from nltk.parse import stanford
from nltk.tree import Tree
import os.path as path
import re
from nltk.corpus import stopwords 
from summarizer.utils.data_helpers import get_parse_info
from summarizer.utils.data_helpers import prune_phrases
dir_path = path.dirname(path.dirname(path.abspath(__file__)))
from nltk.stem.snowball import SnowballStemmer

dir_path = dir_path + '/jars'
os.environ['CLASSPATH'] = dir_path
print(dir_path)
parser = stanford.StanfordParser(model_path="%s/englishPCFG.ser.gz" % (dir_path))
#parser = stanford.StanfordParser(model_path="%s/germanPCFG.ser.gz" % (dir_path), encoding='utf-8')

#"Prospects were dim for resolution of the political crisis in Cambodia in October 1998.",
sents = ["Prime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.",
"King Sihanouk declined to chair talks in either place.",
"A U.S. House resolution criticized Hun Sen's regime while the opposition tried to cut off his access to loans.2",
"But in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.",
"Left out, Sam Rainsy sought the King's assurance of Hun Sen's promise of safety and freedom for all politicians."]

sents = ["Budget negotiations between the White House and House Republicans were delayed on several issues.",
"At issue were provisions that included requiring Federal Health Insurance providers to provide contraceptives to women as Well as a provision to build a road across a wildlife preserve in Alaska.",
"The contraceptive issue faced an uncertain future while Clinton likely will veto the road.",
"There is disagreement also on how to spend the funding on education.",
"This year's budget discussions also have been hampered because it is the first time since budget procedures were established in 1974 that there has been a surplus, preventing agreement on a budget resolution."]

sentences = parser.raw_parse_sents(sents)
language= 'english'
stemmer = SnowballStemmer(language)
stoplist = set(stopwords.words(language))
    
for sent in sentences:
    phrases = []
    parsestr = unicode(list(sent)[0])
    #print 'Sent:', parsestr
    tokens = Tree.fromstring(parsestr).leaves()
    print tokens  
    hash_pos_tokens, phrases = get_parse_info(parsestr, stemmer, language, stoplist)
    check = prune_phrases(phrases, stoplist, stemmer, language)
    for x in check:
        print(unicode(x))
    print('No. of phrases:', len(check))   