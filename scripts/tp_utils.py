import pickle

import nltk
from nltk.corpus import reuters

from string import punctuation
translator = str.maketrans(' ', ' ', punctuation)
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

def normalize_text(doc):
    tokens = []
    for sent in doc.sents:
        sent = str(sent)
        sent = sent.replace('\r', ' ').replace('\n', ' ')
        lower = sent.lower()
        nopunc = lower.translate(translator)
        words = nopunc.split()
        nostop = [w for w in words if w not in stoplist]
        no_numbers = [w if not w.isdigit() else '#' for w in nostop]
        stemmed = [stemmer.stem(w) for w in no_numbers]
        tokens += stemmed
    return tokens

def tokenize(text):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

import pandas as pd
import numpy as np

import spacy
from spacy.lang.en import English
nlp = English()

nlp.add_pipe(nlp.create_pipe('sentencizer'))
nlp2 = spacy.load('en_core_web_sm')

# 
import liwc
import re
from collections import Counter

LIWC_dictionary = '/home/xhta/Robot/liwc/timeori.dic'
# 
def liwc_parse (textstring):
    parse, cat_names = liwc.load_token_parser(LIWC_dictionary)

    tokens = tokenize(textstring)

    rawtext_counts = Counter(category for token in tokens for category in parse(token))

    fpa = rawtext_counts['focuspast']
    fpr = rawtext_counts['focuspresent']
    ffu = rawtext_counts['focusfuture']
    fto = fpa + fpr + ffu
    if fto > 0:
        return fpa, fpr, ffu, fpa/fto, fpr/fto, ffu/fto
    else:
        return fpa, fpr, ffu, 0, 0, 0
#
#
def spacy_parse0 (textstring):
    doc = nlp2(textstring)
    return spacy_parse (doc)

def spacy_parse (doc):

    npast = 0
    npresent = 0
    nfuture = 0
    for token in doc:
    # past tense
        if ( token.tag_  == 'VBN' and  token.dep_  == 'relcl'):
            npast = npast + 1
        if ( token.tag_  == 'VBD' and  token.dep_  == 'relcl'):
            npast = npast + 1
        if ( token.tag_  == 'VBD' and  token.dep_  == 'ccomp'):
            npast = npast + 1
        if ( token.tag_  == 'VBD' and  token.dep_  == 'ROOT'):
            npast = npast + 1
        if ( token.tag_  == 'VBD' and  token.dep_  == 'aux'):
            npast = npast + 1
        if ( token.tag_  == 'VBD' and  token.dep_  == 'auxpass'):
            npast = npast + 1
        if ( token.tag_  == 'VBD' and  token.dep_  == 'conj'):
            npast = npast + 1
        if ( token.tag_  == 'VBN' and  token.dep_  == 'ccomp'):
            npast = npast + 1
    #
    # present tense
        if ( token.tag_  == 'VBZ' and  token.dep_  == 'ROOT'):
            npresent = npresent + 1

        if ( token.tag_  == 'VBP' and  token.dep_  == 'ROOT'):
            npresent = npresent + 1

        if ( token.tag_  == 'VBZ' and  token.dep_  == 'aux'):
            npresent = npresent + 1

        if ( token.tag_  == 'VBZ' and  token.dep_  == 'ccomp'):
            npresent = npresent + 1

        if ( token.tag_  == 'VBP' and  token.dep_  == 'auxpass'):
            npresent = npresent + 1

        if ( token.tag_  == 'VBP' and  token.dep_  == 'aux'):
            npresent = npresent + 1

        if ( token.tag_  == 'VBP' and  token.dep_  == 'ccomp'):
            npresent = npresent + 1

        if ( (token.lemma_ not in ['shall', 'will']) and token.tag_  == 'MD' and token.dep_.strip() == 'aux'):
            npresent = npresent + 1

        if ( (token.lemma_ not in ['shall', 'will']) and token.tag_  == 'VBZ' and token.dep_.strip() == 'ROOT'):
            npresent = npresent + 1

    # future
        if ( (token.lemma_ in ['shall', 'will']) and token.tag_  == 'MD' and token.dep_.strip() == 'aux'):
            nfuture = nfuture + 1

    ntotal = npast + npresent + nfuture
 
    if ntotal > 0:
        return npast, npresent, nfuture, npast/ntotal, float(npresent)/float(ntotal), float(nfuture)/float(ntotal)
    else:
        return npast, npresent, nfuture, 0, 0, 0

import spacy    #  initializing data structures used later
from spacy.lang.en import English
nlp = English()

# input : a long text string
# output: a list of sentences with tokens containing %*&~^$£§=<>\+ removed, \n removed
def clean_text_and_sentencize (textstring):
    textstring = textstring.replace("\n", "")

    textstring = textstring.translate(None, '%*&~^$£§=<>\+@#')

    doc = nlp(textstring)
#
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences


from spacy.lang.en import English
from spacy.matcher import Matcher
#
nlp2 = spacy.load('en_core_web_sm')

dpat1 = [ 
    {'TAG': 'NN', 'DEP:': 'nsubj'},
    {'LOWER': 'shall', 'TAG': 'MD', 'DEP:': 'aux'},
    {'TAG': 'VB', 'DEP:': 'advcl'}
    ]

dpat1n = [ 
    {'TAG': 'NN', 'DEP:': 'nsubj'},
    {'LOWER': 'shall', 'TAG': 'MD', 'DEP:': 'aux'},
    {'LOWER': 'not' },
    {'TAG': 'VB', 'DEP:': 'advcl'}
    ]

dpat2 = [ 
    {'TAG': 'NN', 'DEP:': 'nsubj'},
    {'LOWER': 'shall'},
    {'TAG': 'VBG', 'DEP':'ROOT'}
    ]

dpat3 = [ 
    {'TAG': 'NN', 'DEP:': 'nsubj'},
    {'LOWER': 'shall'},
    {'LOWER': 'not' },
    {'TAG': 'VB', 'DEP':'ROOT'}
    ]

dpat4 = [ 
    {'TAG': 'NN', 'DEP:': 'nsubj'},
    {'LOWER':'shall', 'TAG': 'MD', 'DEP:': 'aux'},
    {'LOWER': 'not', 'OP:': '?'},
    {'TAG': 'VB', 'DEP:': 'auxpass', 'OP': '?'},
    {'TAG': 'VB', 'DEP:': 'ROOT'}
    ]

dmatcher = Matcher(nlp2.vocab)
dmatcher.add('D_PATTERN', None, dpat1, dpat1n, dpat2, dpat3, dpat4)

def deont_parse(textstring):
    doc = nlp2(textstring)
    dmatches = dmatcher (doc)
    return len(dmatches)

def findd (textstring):
    print ("----------------------------------------------------:")
    #print ("textstring:", textstring)
    doc = nlp2(textstring)
    dmatches = dmatcher (doc)
    print ("number of matches:", len(dmatches))
    for match_id, start, end in dmatches:
#    string_id = nlp2.vocab.strings(match_id)
        dmatched_span = doc[start:end]
        print(dmatched_span.text)

def pta (textstr):
    print ("----------------------------------------------------:")
    print ("textstring:", textstr)
    doc = nlp2(textstr)
    for token in doc:
        print(token.text, '\t', token.lemma_, '\t', token.pos_, '\t', "pos_:", token.pos_, '\t', "tag_:", token.tag_, '\t', "dep_:", token.dep_)


def findd2 (textstr):
    print ("----------------------------------------------------:")
    print ("textstring:", textstr)
    doc = nlp2(textstr)
    for token in doc:
        print(token.text, '\t', token.lemma_, '\t', token.pos_, '\t', "pos_:", token.pos_, '\t', "tag_:", token.tag_, '\t', "dep_:", token.dep_)
#
    matches = matcher (doc)
    print ("number of matches:", len(matches))
    for match_id, start, end in matches:
#    string_id = nlp2.vocab.strings(match_id)
        matched_span = doc[start:end]
        print(matched_span.text)

nlp2 = spacy.load('en_core_web_sm')
mpat1 = [ 
        {"LOWER": "would",  "TAG":"MD"},
        {"LOWER": "not",  "OP":"?"},
        {"TAG": "RB", "OP": "*"}, # optional: match 0 or more time
        {"TAG": "VB"} # 
    ] 
mpat1n = [ 
        {"LOWER": "would",  "TAG":"MD"},
        {"TAG": "RB", "OP": "*"}, # optional: match 0 or more time
        {"TAG": "VB"} # 
    ] 
mpat2 = [ 
        {"LOWER": "could",  "TAG":"MD"},
        {"LOWER": "not",  "OP":"?"},
        {"TAG": "RB", "OP": "*"}, # optional: match 0 or more time
        {"TAG": "VB"} # 
    ] 
mpat3 = [ 
        {"LOWER": "might",  "TAG":"MD"},
        {"TAG": "RB", "OP": "*"}, # optional: match 0 or more time
        {"TAG": "VB"} # 
    ] 
mpat3b= [ 
        {"LOWER": "might",  "TAG":"MD"},
        {"LOWER": "not",  "OP":"?"},
        {"TAG": "RB", "OP": "*"}, # optional: match 0 or more time
        {"TAG": "VB"} #
    ] 
#    
mmatcher = Matcher(nlp2.vocab)
mmatcher.add('M_PATTERN', None, mpat1, mpat2, mpat3b)

def modal_parse(textstr):
    doc = nlp2(textstr)
    mmatches = mmatcher (doc)
    return (len(mmatches))

def findm (textstr):
    print ("----------------------------------------------------:")
    print ("textstring:", textstr)
    doc = nlp2(textstr)
    mmatches = mmatcher (doc)
    print ("number of matches:", len(mmatches))
    for match_id, start, end in mmatches:
#    string_id = nlp2.vocab.strings(match_id)
        mmatched_span = doc[start:end]
        print(mmatched_span.text)


