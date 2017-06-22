from pyclausie import ClausIE
from nltk.tree import *
from nltk.draw import tree
from collections import defaultdict
import re, string
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english')).union(set(string.punctuation))

def collect_props(list_props):
    pruned_list = []
    for items in list_props:
        for prop1 in items:
            '''
            if pruned_list:
                flag = 0
                for prop2 in pruned_list:
                    if re.search(prop2, prop1):
                        flag = 1
                if flag == 0:
                    pruned_list.append(prop1)
            else:
            '''
            pruned_list.append(prop1)
    return list(set(pruned_list))    

def flatten_childtrees(trees):
    children = []
    for t in trees:
        if t.height() < 3:
            children.extend(t.pos())
        elif t.height() == 3:
            children.append(Tree(t.node, t.pos()))
        else:
            children.extend(flatten_childtrees([c for c in t]))
    return children

def flatten_deeptree(tree):
    return Tree(tree.node, flatten_childtrees([c for c in tree]))

def prune_props(prop_list):
    prop_list = sorted(prop_list, key=len, reverse=True)
    print list(set(prop_list))
    return prop_list

def create_parsesents(props, sents):
    parse_sents = []
    print('Creating Parse sentences:')
    for index in range(len(props)):
        prop_list, sent = props[index+1], sents[index]
        prop_list = prune_props(prop_list)
        for prop in prop_list:
            if prop.lower() in stoplist:
                continue
            else:
                print prop
                sent = re.sub(" %s " % (prop), ' (C %s) ' % prop, sent)
        sent = re.sub("\) ([^(]+) \(",") (O \\1) (",sent)

        sent = re.sub(r'\s+',' ', sent)
        sent = sent.strip()
        
        bracket_index = sent.rfind(')') + 1
        if bracket_index < len(sent):
            sent = "%s (O %s)" % (sent[:bracket_index], sent[bracket_index+1:])
        bracket_index = sent.find('(')    
        if bracket_index >0:
            sent = "(O %s) %s" % (sent[:bracket_index], sent[bracket_index:])
        
        print 'Before:', sent
        #additional conditions for others
        sent = re.sub("\(C ([^)(]+) \(", "(C (O \\1) (", sent)
        sent = re.sub("\) ([^()]+)\)", ") (C \\1))", sent)
        
        
        sent = "(S %s)" % (sent)
        print index+1, sent
        parse_sents.append(sent)
    return parse_sents

def error_correct(subj, pred, obj):
    items = [subj, pred, obj]
    new_items = []
    for item in items:
        if item.startswith('u '):
            item = item.replace('u ', '')
        if item.endswith(' u'):
            item = item.replace(' u', '')
        new_items.append(item)
    return new_items[0], new_items[1], new_items[2]

def create_trees(sent_triples, sents):
    props = defaultdict(list)
    prev_index = 1
    list_props =[]
    print('Making a list of props:')
    for index, triples in enumerate(sent_triples):
        index = index +1
        for triple in triples:
            _, subj, pred, obj = int(triple.index), triple.subject, triple.predicate, triple.object
            #Error from clauseIE output
            if subj == "u" or pred== 'u' or obj=='u':
                continue
            else:
                subj, pred, obj = error_correct(subj, pred, obj)
                    
                if prev_index != index:
                    props[prev_index] = collect_props(list_props)
                    print(prev_index, props[prev_index])
                    list_props = []
                list_props.append([subj, pred, obj])
                prev_index = index
    
    props[prev_index] = collect_props(list_props)
    print prev_index, props[prev_index]
    
    parse_sents = create_parsesents(props, sents)
    return parse_sents

def get_props(parse_sents):
    props = defaultdict(list)
    for index, sent in enumerate(parse_sents):
        print sent
        t = Tree.fromstring(sent)
        t.draw()
        for i in Tree.fromstring(sent).subtrees():
            if re.match('C', i.label()) and i.height < 3:
                props[index].append(' '.join(i.leaves()))
    return props

def get_clause_sents(sents):
    triples = []
    for i, sent in enumerate(sents):
        try:
            tmp_triples = cl.extract_triples([sent])
            triples.append(tmp_triples)
        except:
            print('Error: failed for line %s' % (sent))
            continue
    parse_sents = create_trees(triples, sents)
    props = get_props(parse_sents)
    for index in props:
        print props[index]


def read_file_tree(filename):
    with open(filename, 'r') as fp:
        parse_lines = fp.read().splitlines()
        get_props(parse_lines) 
def read_file(filename):
    with open(filename, 'r') as fp:
        sents = fp.read().splitlines()
    return sents
    
if __name__ == "__main__":
    '''
    file_name = '/home/local/UKP/avinesh/workspace/casum_summarizer/summarizer/data/processed/DUC2004/d31013t/docs.props/APW19981109.0728'
    read_file_tree(file_name)
    '''
    
    language = 'english'
    if language == 'english':
        cl = ClausIE.get_instance()
    #sents = ["There is speculation the sniper's timing is linked to Remembrance Day because some anti-abortion activists use the day to commemorate aborted fetuses."]
    filename = '/home/local/UKP/avinesh/workspace/casum_summarizer/summarizer/data/processed/DUC2004/d30003t/docs/APW19981020.0241'
    sents = read_file(filename)
    sents = ['His lawyer , Clive Nicholls , said that if a bid to extradite the general succeeded , by the same token Queen Elizabeth II could be extradited to Argentina to face trial for the death of Argentine soldiers in the Falklands war in 1982 .']
    get_clause_sents(sents)
    
        