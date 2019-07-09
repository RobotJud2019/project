import numpy as np
import pandas as pd

import sys
import os
from os import listdir
from os.path import isfile, join
import csv

import shelve
import matplotlib.pyplot as plt

from datetime import datetime
from datetime import timedelta

td = timedelta (days = 0, hours =  0, minutes =  0) # time diff between laptop and realtime
Anfang = datetime.now()
Start = Anfang + td

import pickle

import random
random.seed (1234)

import liwc
import re
from collections import Counter

LIWC_dictionary = '/home/xhta/Robot/liwc/timeori.dic' 	##   file cotaining the LIWC categories, need to set before run

parse, cat_names = liwc.load_token_parser(LIWC_dictionary)

from tp_utils import *

from string import punctuation
translator = str.maketrans(' ', ' ', punctuation)
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')

import spacy
from spacy.matcher import Matcher
nlp3 = spacy.load('en_core_web_sm')

matcher = Matcher(nlp3.vocab)

def normalize_text(doc):	 # 
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

fpath = '/home/xhta/Robot/cases/'	# where the cases are located in file system , need to set before run

sample_size = 0.1   # fraction of the whole corpis used for sampling

cases_metadata_pickle = '/home/xhta/Robot/cases_metadata.20190627_0039.pkl'	# where the cases metadata are located in file system , need to set before run

import pickle
df = pickle.load(open(cases_metadata_pickle, "rb"))
#df = df.sample(frac=sample_size)

ldir = listdir(fpath)

import spacy	#  initializing data structures used later
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

nlp2 = spacy.load('en_core_web_sm')

n_sandglass = 29 	# displays on console, useful for long runs
cp_interval = 17      # checkpoint time interval

i = 0
for fname in ldir:	# read in the document samples and populate the dataframe with linguistic features
    if isfile(fname) == 'False': continue
    lae = len(fname)
    cname = fname[5:(lae-4)]
    year = fname[0:4]

    if (not (cname in df.index)): continue # useful if only using a sample of cases

    if ( i% n_sandglass==0):
        print((datetime.now() + td).strftime('%Y%m%d_%H:%M:%S'), "  i:", i, "  running script:", sys.argv[0])	 
#
    fna2 = join(fpath, year + '_' + cname + '.txt')
    rawtext = open(fna2).read()
    rawtext = rawtext.replace('\n', '')
    rawtext = rawtext.replace("\'s", "'s")

    doc = nlp2(rawtext)
    df.at[cname, 'nlets'] =  len(rawtext)
    sentences = [sent.string.strip() for sent in doc.sents]
    df.at[cname, 'nsents'] =  len(sentences)
    df.at[cname, 'nwords'] = len([token for token in doc if not token.is_punct])

    df.at[cname, 'doc'] = rawtext
#
    cltoks = normalize_text(doc)
    ntoks = [str(token).lower() for token in list(doc) if (not token.is_punct) & (not token.is_space) & (not token.is_stop) & (str(token) in cltoks)]
#
    df.at[cname, 'ntoks'] = len(ntoks)

    df.at[cname, 'nverbs'] = len([w for w in list(doc) if w.tag_.startswith('V')])

    npast, npresent, nfuture, antpast, antpresent, antfuture = spacy_parse(doc)		# get the past, present, and future tenses based on spacy's tagging
#
    df.loc[cname, 'npast'] = npast			# npast, npresent and nfuture are numbers indicating the frequencies of past, present and future tenses in the doc
    df.loc[cname, 'npresent'] = npresent
    df.loc[cname, 'nfuture'] = nfuture

    df.loc[cname, 'antpast'] = antpast			# antpast = npast / (npast + npesent + nfuture)
    df.loc[cname, 'antpresent'] = antpresent
    df.loc[cname, 'antfuture'] = antfuture

    nfpast, nfpresent, nffuture, antfpast, antfpresent, antffuture = liwc_parse(rawtext)	# get the past, present and future focus using LIWC
#
    df.loc[cname, 'nfpast'] = nfpast
    df.loc[cname, 'nfpresent'] = nfpresent
    df.loc[cname, 'nffuture'] = nffuture

    df.loc[cname, 'antfpast'] = antfpast		# antfpast = nfpast / (nfpast + nfpresent + nffuture)
    df.loc[cname, 'antfpresent'] = antfpresent
    df.loc[cname, 'antffuture'] = antffuture

    lse = nltk.sent_tokenize(rawtext)			# break up the rawtext into a list of sentences and parse the sentences individually
    ldeont = 0						# to find the frequencies of deontic futures and of modal verbs (would could might)
    lmodal = 0
    for j in range(len(lse)):				# both deont_parse and modal_parse are spacy-based and located in tp_utils.py
        ldeont += deont_parse(lse[j])
        lmodal += modal_parse(lse[j])
    df.at[cname, "ldeont"] = ldeont
    df.at[cname, "lmodal"] = lmodal
#
    if (i % cp_interval  == 0):				# checkpointing in intervals
        je = datetime.now() + td
        pkl_fname = 'pj_df_full.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
        print ("jetzt:{0}  i:{1} ... intermediate checkpointing to {2}".format(je, i, pkl_fname))
        df.to_pickle(pkl_fname)
    i = i + 1

je = datetime.now() + td
pkl_fname = 'pj_df_full.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
print ("jetzt:{0}  i:{1} ... final write to pickle {2}".format(je, i, pkl_fname))
df.to_pickle(pkl_fname)




