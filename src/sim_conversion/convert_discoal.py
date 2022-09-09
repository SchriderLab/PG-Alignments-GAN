# #!/usr/bin/python
import sys
import numpy as np
#import h5py
import pandas as pd
import os
import random

####### usage: cat <msfile> | python convert_discoal.py outdir #######

def split_sites(word):
    return [int(char) for char in word]

nrep = 1
curr_rep = []
nsam = 64  ####### number of individuals in each rep
nsites = 64
sel_pos = 0.5
outdir = sys.argv[1]
run_convert = 0
skip_sim = True

def normalize_positions(pos, start, end):
    width = end - start
    norm_pos = (pos - pos.min())/pos.ptp() * width + start
    norm_pos = np.round(norm_pos,6)
    return norm_pos

if not os.path.isdir(outdir):
    os.mkdir(outdir)

for line in sys.stdin:
    line = line.strip()
    if line.startswith( "segsites" ):
        rep_sites = int(line.split(" ")[1])
        if rep_sites < nsites:
            skip_sim = True
            continue
        else:
            skip_sim = False
            #start_site = int((rep_sites - nsites) / 2)
            #end_site = start_site + nsites
            continue
    if line.startswith( "position" ):
        if not skip_sim:
            pos = line.split(" ")
            pos = [float(x) for x in pos[1:]]
            pos = np.asarray(pos)
            sel_idx = (np.abs(pos - sel_pos)).argmin()
            if int((sel_idx + 1) - (nsites / 2)) >= 0 and int((sel_idx + 1) - (nsites / 2) + nsites) <= rep_sites:
                run_convert = 1
                rep_sam = 0
                start_site = int((sel_idx + 1) - (nsites / 2))
                end_site = int(start_site + nsites)
                
                if start_site == 0:
                    pos.insert(0,0)
                
                if end_site == len(pos):
                    pos.append(1)

                pos = pos[start_site-1:end_site+1]
                pos = normalize_positions(pos,0,1)
                pos = pos[1:len(pos)-1]
                pos = pd.DataFrame(pos)
                pos.to_csv(outdir+str(nrep)+'_pos.csv',header=False,index=False)
            continue
        else:
            continue
    if run_convert == 1:
        if rep_sam < nsam:
            curr_rep.append(split_sites(line)[start_site:end_site])
            rep_sam +=1
        if rep_sam == nsam:
            curr_rep = pd.DataFrame(curr_rep)
            #with h5py.File(outdir+str(nrep)+'.h5', 'w') as hf:
            #    hf.create_dataset(str(nrep), data=curr_rep)
            curr_rep.to_csv(outdir+str(nrep)+'_sites.csv',header=False,index=False)
            #print(curr_rep)
            rep_sam = 0
            nrep += 1
            run_convert = 0
            curr_rep = []
print(nrep - 1)
