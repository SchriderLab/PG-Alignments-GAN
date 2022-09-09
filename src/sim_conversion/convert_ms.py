# #!/usr/bin/python
import sys
import numpy as np
#import h5py
import pandas as pd
import os

####### usage: cat <msfile> | python convert_ms.py odir #######

def split_sites(word):
    return [int(char) for char in word]

nrep = 1
curr_rep = []
set_nsam = 64  ####### number of individuals in each rep
outdir = sys.argv[1]
run_convert = 0

if not os.path.isdir(outdir):
    os.mkdir(outdir)

for line in sys.stdin:
    line = line.strip()
    if line.startswith( "position" ):
        run_convert = 1
        nsam = 0
        pos = line.split(" ")
        pos = pd.DataFrame(pos[1:])
        pos.to_csv(outdir+str(nrep)+'_pos.csv',header=False,index=False)
        continue
    if run_convert == 1:
        if nsam < 64:
            curr_rep.append(split_sites(line))
            nsam +=1
        if nsam == 64:
            curr_rep = pd.DataFrame(curr_rep)
            #with h5py.File(outdir+str(nrep)+'.h5', 'w') as hf:
            #    hf.create_dataset(str(nrep), data=curr_rep)
            curr_rep.to_csv(outdir+str(nrep)+'_sites.csv',header=False,index=False)
            #print(curr_rep)
            nsam = 0
            nrep += 1
            run_convert = 0
            curr_rep = []

