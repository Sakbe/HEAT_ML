import glob
import numpy as np

dirs = glob.glob('/scratch/gpfs/dc2313/HEAT/data/sparc_000000_eq*')
dirs.sort()

ShadowMasks = []

for d in dirs:

    csv_path = d + '/000001/shadowMask_all.csv'
    try:
        data = np.loadtxt(csv_path, delimiter=',',skiprows=1)  
        ShadowMasks.append(data[:,3])
    except IOError:
        print(f"File not found or unable to read: {csv_path}")
        
ShadowMasks=np.array(ShadowMasks)
XYZ= data[:,0:3]

np.savez("ShadowMasks_all.npz",XYZ=XYZ,ShadowMasks=ShadowMasks)


