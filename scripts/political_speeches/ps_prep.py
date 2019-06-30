# This script prepares the metadata of the political speeches.

import pandas as pd
import os
import fileinput
df = pd.DataFrame(index=range(0,0), columns = ['fname', 'title', 'location', 'date', 'comment'], dtype = object)


fpath = '/home/xhta/Robot/proj/posp/speeches/raw'	# must be set before run
from os import listdir
ldir = listdir(fpath)
a = 0
b = 0
fo = open("dummy.txt", "w")
for fname in ldir:
    print(fname)
    for line in fileinput.input(os.path.join(fpath, fname), openhook=fileinput.hook_encoded("iso-8859-1")):
        if (line.find('<docid') >= 0): 
            curr = line[7:-2].replace(' ','').replace('id','')
            df.loc[curr, 'fname'] = fname
            if (fo.closed == False):
                fo.close()
            fo = open(curr + ".txt", "w")
        elif (line.find('<title') >= 0): df.loc[curr, 'title'] = line[7:-2]
        elif (line.find('<location') >= 0): df.loc[curr, 'location'] = line[9:-2]
        elif (line.find('<date') >= 0): df.loc[curr, 'date'] = line[6:-2]
        elif (line.find('<comment') >= 0): df.loc[curr, 'comment'] = line[9:-2]
#        else: b = b + 1
        else: fo.write(line)
    fo.close()
    fileinput.close()

for inde in df.index:
    if 'Obama' in df.loc[inde, 'fname']: df.at[inde, 'Speaker'] = 'Obama'
    if 'McCain' in df.loc[inde, 'fname']: df.at[inde, 'Speaker'] = 'McCain'
df2 = pd.DataFrame(data=df, columns = ['Speaker', 'date', 'fname', 'title', 'location', 'comment'])
df2 = df2.sort_index()
df2.to_csv("posp_metadata.csv", sep = ';')
