# see blog.chapagain.com.np/python-twittet-sentiment-analysis-on

import pickle 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import string

from datetime import datetime
from datetime import timedelta

td = timedelta (days = 0, hours = 6 , minutes = 48)  # difference between laptop time and realtime
Anfang = datetime.now()
Start = Anfang + td

from tp_utils import *

import preprocessor as prp

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

from nltk.corpus import twitter_samples
fids = twitter_samples.fileids()
strs = twitter_samples.strings(fids[2])
#len(strs) # 20000

def clean_tweet(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    tweet = re.sub(r'['+string.punctuation+']+', '', tweet)
    return tweet

def clean_tweet2(tweet):
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r' #[\w]+ ', ' ', tweet)
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
#    tweet = re.sub(r'['+string.punctuation+']+', '', tweet)
    tweet = re.sub(r' ['+string.punctuation+']+ ', ' ', tweet)
    return tweet

import spacy
from spacy.lang.en import English
nlp = English()

nlp2 = spacy.load('en_core_web_sm')

dftw = pd.DataFrame(index = range(0,0), columns=['leng', 'nwords', 'ntoks', 'nverbs', 'npast', 'npresent', 'nfuture', 'nfpast', 'nfpresent', 'nffuture', 'tw'], dtype = int)

for col in ['leng', 'nwords', 'ntoks', 'nverbs', 'npast', 'npresent', 'nfuture', 'nfpast', 'nfpresent', 'nffuture']:
    dftw[col] = dftw[col].astype(int)

dftw['tw'] = dftw['tw'].astype(object)

twfids = twitter_samples.fileids()
strs = twitter_samples.strings(twfids[2])

for i in range(len(strs)): 
    twc = prp.clean(strs[i])
    dftw.at[i, 'tw'] = twc

    dftw.at[i, 'leng'] = len(twc) 

    doc = nlp2(twc)

    cltoks = normalize_text(doc)
    ntoks = [str(token).lower() for token in list(doc) if (not token.is_punct) & (not token.is_space) & (not token.is_stop) & (str(token) in cltoks)]
    dftw.at[i, 'ntoks'] = len(ntoks)

    sentences = [sent.string.strip() for sent in doc.sents]
    dftw.at[i, 'nsents'] =  len(sentences)
    dftw.at[i, 'nwords'] = len([token for token in doc if not token.is_punct])

    dftw.at[i, 'nverbs'] = len([w for w in list(doc) if w.tag_.startswith('V')])

    npast, npresent, nfuture, antpast, antpresent, antfuture = spacy_parse(doc)
#
    dftw.at[i, 'npast'] = npast
    dftw.at[i, 'npresent'] = npresent
    dftw.at[i, 'nfuture'] = nfuture

    dftw.at[i, 'antpast'] = antpast
    dftw.at[i, 'antpresent'] = antpresent
    dftw.at[i, 'antfuture'] = antfuture

    nfpast, nfpresent, nffuture, antfpast, antfpresent, antffuture = liwc_parse(twc)
#
    dftw.at[i, 'nfpast'] = nfpast
    dftw.at[i, 'nfpresent'] = nfpresent
    dftw.at[i, 'nffuture'] = nffuture
 
    dftw.at[i, 'antfpast'] = antfpast
    dftw.at[i, 'antfpresent'] = antfpresent
    dftw.at[i, 'antffuture'] = antffuture

    dftw.at[i, 'ldeont'] = deont_parse(twc)
    dftw.at[i, 'lmodal'] = modal_parse(twc)
#
    if (i %579 == 0):
        je = datetime.now() + td
        pkl_fname = 'pj_dftw_full.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
        print ("jetzt:{0}  i:{1} ... interim checkpointing to {2}".format(je, i, pkl_fname))
        dftw.to_pickle(pkl_fname)

je = datetime.now() + td
pkl_fname = 'pj_dftw_full.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
print ("jetzt:{0}  i:{1} ... final checkpointing to {2}".format(je, i, pkl_fname))
dftw.to_pickle(pkl_fname)


 
