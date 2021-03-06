
### cases corpus:
* ```rmd2.py``` or ```rmd2.ipynb```: reads the cases metadata and judges metadata, merges them and writes to pickle
* ```docases``` : reads the cases documents, extracts the BOW and the linguistic featues and writes to pickle
* ```cases_analysis.ipynb``` : interactive analysis of the cases pickles, plot histograms, time series, perform stat tests


### political speeches:
* ```ps_prep.py``` or ```ps_prep.ipypb```: reads the source files, extracts the metadata in the header and generates ```posp_metadata.csv```
* ```doposp.py``` or ```doposp.ipynb```: reads the source files, extracts the BOW and the linguistic features and writes to pickle




### reuters:
* ```doreu.py``` or ```doreu.ipynb```: processes the reuters dataset in nltk.corpus, extracts the BOW and the linguistic features and writes to pickle. Displays a plot at the end.



### twitter:
* ```dotw.py``` or ```dotw.ipynb```: processes the twitter dataset in nltk.corpus, extracts the BOW and the linguistic features and writes to pickle. Displays a plot at the end.

#### A Jupyter notebook and a python script having the same name have the same functionalities.

#### For a graphical overview of the data flow see Flow.pdf.

### pickles in subfolder "data": 
* cases corpus:			```pj_df_full.20190629_095112.pkl``` (pickle uploaded to polybox)
* political speeches:		```pj_demo_dfmo_full.20190629_163240.pkl```	(rawtext uploaded to polybox)
* reuters data from the nltk corpus : ```pj_dfre_full.20190629_115819.pkl```
* twitter data from the nltk corpus : ```pj_dftw_full.20190630_005652.pkl```

### metadata (= column names as seen in DataFrame.columns)
* nlets or leng : length of the document 
* nsents : number of sentences
* nwords : number of words in the document (punctuations excluded)
* ntoks : number of tokens (= words - stopwords - spaces)

* npast, npresent, nfuture : number of verbs in past tense, present tense, future tense found by POS tagging
   - antpast = npast / (npast + npresent + nfuture)  , antpresent, antfuture analogous
   - nfpast, nfpresent, nffuture : number of words with focus past, present, future found by LIWC
   - antfpast = nfpast / (nfpast + nfpresent + nffuture)  , antfpresent, antffuture analogous

* ldeontic : the number of deontic futures found in the document
  - deontic_ratio : ldeontic / nwords

* lmodal : the number of verbs in modal form (would could might)

Deontic futures can be found by using the function ```findd``` in the module tp_utils.
The results are as good as the rules one defines.
A total of 3271 deontic futures were found in 1592 documents in the cases corpus.
<br>
Example

```
df = pickle.load(open("pj_df_full.20190629_095112.pkl", "rb"))
findd(df.loc["X3IMHM", "doc"])
number of matches: 10
chapter shall have
work shall have
recovery shall be
city shall fail
property shall be
lien shall not attach
property shall be
improvement shall be
council shall be
assessment shall become
```

rules to find the linguistis features are defined in ```tp_utils.py```: ```spacy_parse, LIWC_parse, deontic_parse, modal_parse```

```
source code tested using python 3.5
        	pandas		0.24.2
		numpy   	1.16.2
		scipy   	1.3.0
		matplotlib 	2.2.2
		liwc   		0.4.1
		spacy   	2.0.18
		nltk   		3.3
		tweet-preprocessor 0.5.0
	for analysis 
		statsmodels	0.9.0    (attention: conflicts with scipy 1.3.0, works with scipy 1.2.0)
		linearmodels	4.12
		skll		1.5.3
		xgboost		0.90
		skater		1.1.2	
		textblob	0.15.3
		seaborn		0.9.0
````

