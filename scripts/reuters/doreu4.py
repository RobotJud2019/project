# this python script processes the reuters dataset from the nltk corpus

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta

td = timedelta (days = 0, hours =  0 , minutes = 53)   # diff between laptop time and realtime
Anfang = datetime.now()
Start = Anfang + td

import pickle

from tp_utils import *

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

import liwc
import re
from collections import Counter

LIWC_DICTIONARY = '/home/xhta/Robot/liwc/timeori.dic'
parse, cat_names = liwc.load_token_parser(LIWC_DICTIONARY)

def tokenize(text):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

import spacy
from spacy.lang.en import English
nlp = English()

nlp.add_pipe(nlp.create_pipe('sentencizer'))
nlp2 = spacy.load('en_core_web_sm')

dfre = pd.DataFrame(index = range(0,0), columns=['cats', 'tetr', 'leng', 'nwords', 'ntoks', 'nverbs', 'npast', 'npresent', 'nfuture', 'nfpast', 'nfpresent', 'nffuture', 'rt'], dtype = int)

for col in [ 'tetr', 'leng', 'nwords', 'ntoks', 'nverbs', 'npast', 'npresent', 'nfuture', 'nfpast', 'nfpresent', 'nffuture']:
    dfre[col] = dfre[col].astype(int)

dfre['cats'] = dfre['cats'].astype(object)
dfre['rt'] = dfre['rt'].astype(object)

fids = reuters.fileids()
i = 0
for fid in fids:
    rt = reuters.raw(fid)
    rtc = rt.replace("\n", "")
    rtc = rtc.replace("\'s", "'s")
    dfre.at[fid, 'rt'] = rtc 

    dfre.at[fid, 'leng'] = len(rtc) 
    dfre.at[fid, 'cats'] = reuters.categories(fid)
    if "test" in fid:
        dfre.at[fid, 'tetr'] = 0
    else: dfre.at[fid, 'tetr'] = 1

    doc = nlp2(dfre.loc[fid, 'rt'])

    cltoks = normalize_text(doc)
    ntoks = [str(token).lower() for token in list(doc) if (not token.is_punct) & (not token.is_space) & (not token.is_stop) & (str(token) in cltoks)]
    dfre.at[fid, 'ntoks'] = len(ntoks)

    sentences = [sent.string.strip() for sent in doc.sents]   # to do : replace by nltk.sent_tokenize ? might speed up
    dfre.at[fid, 'nsents'] =  len(sentences)
    dfre.at[fid, 'nwords'] = len([token for token in doc if not token.is_punct])

    dfre.at[fid, 'nverbs'] = len([w for w in list(doc) if w.tag_.startswith('V')])

    npast, npresent, nfuture, antpast, antpresent, antfuture = spacy_parse(doc)
#
    dfre.loc[fid, 'npast'] = npast
    dfre.loc[fid, 'npresent'] = npresent
    dfre.loc[fid, 'nfuture'] = nfuture

    dfre.loc[fid, 'antpast'] = antpast
    dfre.loc[fid, 'antpresent'] = antpresent
    dfre.loc[fid, 'antfuture'] = antfuture

    nfpast, nfpresent, nffuture, antfpast, antfpresent, antffuture = liwc_parse(rtc)
#
    dfre.loc[fid, 'nfpast'] = nfpast
    dfre.loc[fid, 'nfpresent'] = nfpresent
    dfre.loc[fid, 'nffuture'] = nffuture

    dfre.loc[fid, 'antfpast'] = antfpast
    dfre.loc[fid, 'antfpresent'] = antfpresent
    dfre.loc[fid, 'antffuture'] = antffuture

    dfre.loc[fid, 'ldeont'] = deont_parse(rtc)
    dfre.loc[fid, 'lmodal'] = modal_parse(rtc)
#
    if (i %579 == 0):
        je = datetime.now() + td
        pkl_fname = 'pj_dfre_full.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
        print ("jetzt:{0}  i:{1} ... interim checkpointing to {2}".format(je, i, pkl_fname))
        dfre.to_pickle(pkl_fname)
    i = i + 1

je = datetime.now() + td
pkl_fname = 'pj_dfre_full.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
print ("jetzt:{0}  ... final checkpointing to {1}".format(je, pkl_fname))
dfre.to_pickle(pkl_fname)

sAvepast = dfre["antpast"].mean()
sAvepresent = dfre["antpresent"].mean()
sAvezfuture = dfre["antfuture"].mean()

lAvepast = dfre["antfpast"].mean()
lAvepresent = dfre["antfpresent"].mean()
lAvezfuture = dfre["antffuture"].mean()

modfplot = pd.DataFrame({'Avepast' : [lAvepast, sAvepast], 'Avepresent': [lAvepresent, sAvepresent], 'Avezfuture': [lAvezfuture, sAvezfuture]}, index =
 ['LIWC', 'Spacy'] )
modfplot.plot.bar(rot=0)
plt.show()

