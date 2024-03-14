#test script to get all the data


import glob
import re
from read_efit import read_efit 
import numpy as np

files = glob.glob('/scratch/gpfs/rmc2/HEAT/data/sparc*eq*')

def extract_number(filename):
    # Extracts the last number from the filename
    return int(re.findall(r'\d+$', filename)[-1])

sorted_files = sorted(files, key=extract_number)

Bt0s = []; Ips = []; q95s = []; alpha1s = []; alpha2s = []
for i,fi in enumerate(sorted_files):
    Bt0, Ip, q95, alpha1, alpha2  = read_efit(fi+'/000001/g000000.00001')
    Bt0s.append(Bt0)
    Ips.append(Ip)
    q95s.append(q95)
    alpha1s.append(alpha1)
    alpha2s.append(alpha2)
    
Bt0s = np.array(Bt0s)
Ips = np.array(Ips)
q95s = np.array(q95s)
alpha1s = np.array(alpha1s)
alpha2s = np.array(alpha2s)

np.savez('efits_all.npz',Bt0s=Bt0s, Ips=Ips, q95s=q95s, alpha1s=alpha1s, alpha2s=alpha2s)
