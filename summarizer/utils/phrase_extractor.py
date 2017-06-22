import nltk
from nltk.corpus import stopwords

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP' or t.label()=='VP'):
        yield subtree.leaves()

def normalise(word, stemmer):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word = stemmer.stem(word)
    #word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word, stopwords):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted

def get_terms(tree, stemmer, stopwords):
    for leaf in leaves(tree):
        term = [ normalise(w, stemmer) for w,t in leaf if acceptable_word(w, stopwords) ]
        yield term

if __name__ == '__main__':
    text = "The Buddha, the Godhead, resides quite as comfortably in the circuits of a digital computer or the gears of a cycle transmission as he does at the top of a mountain or in the petals of a flower. To think otherwise is to demean the Buddha...which is to demean oneself."
    text = "Budget negotiations between the White House and House Republicans were delayed on several issues."
    # Used when tokenizing words
    sentence_re = r'''(?x)      # set flag to allow verbose regexps
          ([A-Z])(\.[A-Z])+\.?  # abbreviations, e.g. U.S.A.
        | \w+(-\w+)*            # words with optional internal hyphens
        | \$?\d+(\.\d+)?%?      # currency and percentages, e.g. $12.40, 82%
        | \.\.\.                # ellipsis
        | [][.,;"'?():-_`]      # these are separate tokens
    '''
    language= 'english'
    stemmer = nltk.SnowballStemmer(language)
    stoplist = set(stopwords.words(language))
    
    #Taken from Su Nam Kim Paper...
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        VP:
            {<V.*>}  # terminated with Verbs
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...        
    """
    chunker = nltk.RegexpParser(grammar)
    
    toks = nltk.word_tokenize(text)
    
    postoks = nltk.tag.pos_tag(toks)
    
    print postoks
    
    tree = chunker.parse(postoks)   
    
    phrases = get_terms(tree, stemmer, stoplist)
    phrase_list = [ ' '.join(term) for term in phrases if term]
    print phrase_list