# this script is used to read from the cases metadata and enrich them with the judges metadata
# the cases metadata start out wth caseid, case_reversed, judge_id, year and log_cites
# the judges metadata list LastName, FirstName, Gender, CourtType, nominating Presidents and their party affiliation

import pandas as pd
import numpy as np

CASES_MD = "/home/xhta/Robot/case_metadata.csv"
JUDGES_MD = "/home/xhta/Robot/judges/judges-7.csv"

md = pd.DataFrame(index = range(0,0), columns=['caseid','case_reversed','judge_id','year','log_cites'])

md = pd.read_csv (CASES_MD)
md["case_reversed"] = md["case_reversed"].fillna(0).astype(int)
md["judge_id"] = md["judge_id"].fillna(0).astype(int)
md["year"] = md["year"].fillna(0).astype(int)
md["log_cites"] = md["log_cites"].fillna(0).astype(float)

empty_str = '' * len(md)
md["LastName"] = pd.Series(empty_str, index = md.index)
md["FirstName"] = pd.Series(empty_str, index = md.index)
md["Gender"] = pd.Series([-1]*len(md), index = md.index)
md["Pres"] = pd.Series(empty_str, index = md.index)
md["Party"] = pd.Series(empty_str, index = md.index)

md["log_cites"] = md["log_cites"].fillna(0)

for col in ['LastName', 'FirstName', 'Pres', 'Party']:
    md[col] = md[col].astype(object)

md["Gender"] = md["Gender"].astype(int)
#
ju = pd.read_csv(JUDGES_MD, sep=';', encoding='cp1252')
ju.columns = ["judge_id", "LastName", "FirstName", "Gender", "CourtType", "Pres", "Party"]
#
ju["judge_id"] = ju["judge_id"].fillna(0).astype(int)
#
ju.loc[ ju.Gender  == 'Female', 'Gender' ] = 0
ju.loc[ ju.Gender  == 'Male', 'Gender' ] = 1
#

md = md.set_index("caseid")
ju = ju.set_index("judge_id")
#
for inde in md.index:
    jid = md.loc[inde, "judge_id"] 
    if jid not in ju.index: continue
    md.loc[inde, "LastName"] = ju.loc[jid, "LastName"]
    md.loc[inde, "FirstName"] = ju.loc[jid, "FirstName"]
    md.loc[inde, "Gender"] = ju.loc[jid, "Gender"]
    md.loc[inde, "Pres"] = ju.loc[jid, "Pres"]
    md.loc[inde, "Party"] = ju.loc[jid, "Party"]
#
from datetime import datetime
je = datetime.now()
pkl_fname = 'cases_metadata.' + je.strftime('%Y%m%d_%H%M%S' + ".pkl")
md.to_pickle(pkl_fname)
#md.to_pickle("cases_metadata.20190627_0039.pkl")
