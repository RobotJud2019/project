# this script processes the political speeches. Was modified from demo_mo.py
from datetime import datetime
from datetime import timedelta

td = timedelta (days = 0, hours = 0, minutes = 51)	# to compensate time difference between laptop and realtime
Anfang = datetime.now()
Start = Anfang + td

import csv
import pandas as pd
import os
from os import listdir
from os.path import isfile, join

import shelve
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import liwc
import re
from collections import Counter

LIWC_dictionary= '/home/xhta/Robot/liwc/timeori.dic'
POSP_METADATA = '/home/xhta/Robot/proj/posp/posp_metadata.csv'
POSP_CLEANDATA = '/home/xhta/Robot/proj/posp/clean'

parse, cat_names = liwc.load_token_parser(LIWC_dictionary)

def tokenize(text):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

from string import punctuation
translator = str.maketrans(' ', ' ', punctuation)
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

import spacy
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

from spacy.matcher import Matcher
nlp2 = spacy.load('en_core_web_sm')

from tp_utils import *

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

dfmo = pd.DataFrame(index = range(0,0), columns=['jahr', 'leng', 'datum', 'speaker', 'nwords', 'ntoks', 'nverbs', 'npast', 'npresent', 'nfuture', 'nfpast', 'nfpresent', 'nffuture', 'doc'], dtype = int)
dfmo['datum'] = dfmo['datum'].astype('object')

with open (POSP_METADATA) as fcmo:
    readCSV = csv.reader(fcmo, delimiter = ';')
    next(readCSV, None)   # skip 1 line
    for Zei in readCSV:
        dfmo.loc[Zei[0]] = [0, 0, Zei[2], Zei[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

dfmo['doc'] = dfmo['doc'].astype('object')

fpath = POSP_CLEANDATA
ldir = listdir(fpath) 

i = 0
for fname in ldir:
    lae = len(fname)
    cname = fname[0:(lae-4)]
    rawtext = open(os.path.join(fpath, fname), "r").read()
    rawtext = rawtext.replace("\n", '')
    doc = nlp2(rawtext)
#
    dfmo.at[cname, 'nwords'] = len([token for token in doc if not token.is_punct])
    dfmo.at[cname, 'nverbs'] = len([w for w in list(doc) if w.tag_.startswith('V')])
#
    cltoks = normalize_text(doc)
    ntoks = [str(token).lower() for token in list(doc) if (not token.is_punct) & (not token.is_space) & (not token.is_stop) & (str(token) in cltoks)]
    dfmo.at[cname, 'ntoks'] = len(ntoks)
    #print("ntoks:", df.at[cname, 'ntoks'])
#
    laeng =  len(rawtext)
    dfmo.at[cname, 'leng'] = laeng 
#
    tokens = tokenize(rawtext)
#
    dfmo.at[cname, 'doc'] =  rawtext

    npast, npresent, nfuture, antpast, antpresent, antfuture = spacy_parse(doc)
#
    dfmo.loc[cname, 'npast'] = npast
    dfmo.loc[cname, 'npresent'] = npresent
    dfmo.loc[cname, 'nfuture'] = nfuture

    dfmo.loc[cname, 'antpast'] = antpast
    dfmo.loc[cname, 'antpresent'] = antpresent
    dfmo.loc[cname, 'antfuture'] = antfuture

    nfpast, nfpresent, nffuture, antfpast, antfpresent, antffuture = liwc_parse(rawtext)
#
    dfmo.loc[cname, 'nfpast'] = nfpast
    dfmo.loc[cname, 'nfpresent'] = nfpresent
    dfmo.loc[cname, 'nffuture'] = nffuture

    dfmo.loc[cname, 'antfpast'] = antfpast
    dfmo.loc[cname, 'antfpresent'] = antfpresent
    dfmo.loc[cname, 'antffuture'] = antffuture

    lse = nltk.sent_tokenize(rawtext)
    ldeont = 0
    lmodal = 0
    for j in range(len(lse)):
        ldeont += deont_parse(lse[j])
        lmodal += modal_parse(lse[j])
    dfmo.at[cname, "ldeont"] = ldeont
    dfmo.at[cname, "lmodal"] = lmodal
#
    if (i %19 == 0):
        je = datetime.now() + td
        pkl_fname = 'pj_demo_dfmo_full.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
        print ("jetzt:{0}  i:{1} ... interim checkpointing to {2}".format(je, i, pkl_fname))
        dfmo.to_pickle(pkl_fname)
    i = i + 1

je = datetime.now() + td
pkl_fname = 'pj_demo_dfmo_full.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
print ("jetzt:{0}  i:{1} ... final checkpointing to {2}".format(je, i, pkl_fname))
dfmo.to_pickle(pkl_fname)

import matplotlib.pyplot as plt

#focus  liwc
for inde in dfmo.index:
   dfmo.loc[inde, 'totf'] = dfmo.loc[inde, 'nfpast'] + dfmo.loc[inde, 'nfpresent'] + dfmo.loc[inde, 'nffuture']
   dfmo.loc[inde, 'antfpast'] = dfmo.loc[inde, 'nfpast'] / dfmo.loc[inde, 'totf']
   dfmo.loc[inde, 'antfpresent'] = dfmo.loc[inde, 'nfpresent'] / dfmo.loc[inde, 'totf']
   dfmo.loc[inde, 'antffuture'] = dfmo.loc[inde, 'nffuture'] / dfmo.loc[inde, 'totf']

oAvefpast = dfmo.loc[ dfmo['speaker'] == 'Obama', 'antfpast'].mean()
oAvefpresent = dfmo.loc[ dfmo['speaker'] == 'Obama', 'antfpresent'].mean()
oAveffuture = dfmo.loc[ dfmo['speaker'] == 'Obama', 'antffuture'].mean()

mAvefpast = dfmo.loc[ dfmo['speaker'] == 'McCain', 'antfpast'].mean()
mAvefpresent = dfmo.loc[ dfmo['speaker'] == 'McCain', 'antfpresent'].mean()
mAveffuture = dfmo.loc[ dfmo['speaker'] == 'McCain', 'antffuture'].mean()

modfplot = pd.DataFrame({'Avef1past' : [mAvefpast, oAvefpast], 'Avef2present': [mAvefpresent, oAvefpresent], 'Avef3future': [mAveffuture, oAveffuture]}, index = ['McCain', 'Obama'] )
modfplot.plot.bar(rot=0)
plt.show()

# POS taggging spacy
oAvepast = dfmo.loc[ dfmo['speaker'] == 'Obama', 'antpast'].mean()
oAvepresent = dfmo.loc[ dfmo['speaker'] == 'Obama', 'antpresent'].mean()
oAvefuture = dfmo.loc[ dfmo['speaker'] == 'Obama', 'antfuture'].mean()

mAvepast = dfmo.loc[ dfmo['speaker'] == 'McCain', 'antpast'].mean()
mAvepresent = dfmo.loc[ dfmo['speaker'] == 'McCain', 'antpresent'].mean()
mAvefuture = dfmo.loc[ dfmo['speaker'] == 'McCain', 'antfuture'].mean()

modtplot = pd.DataFrame({'Ave1past' : [mAvepast, oAvepast], 'Ave2present': [mAvepresent, oAvepresent], 'Ave3future': [mAvefuture, oAvefuture]}, index = ['McCain', 'Obama'] )
modtplot.plot.bar(rot=0, sort_columns=False)
plt.show()

