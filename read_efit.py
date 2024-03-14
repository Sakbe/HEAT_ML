#reads from an EFIT geqdsk file and returns the inputs parameters to use
#in the machine learning model

import EFIT.equilParams_class as eqcl
import numpy as np

def pitch_angle(ep,R,Z):
    return np.squeeze(np.arctan(ep.BpFunc(R,Z)/np.abs(ep.BtFunc(R,Z)))*180./np.pi)

def read_efit(filename):
    ep = eqcl.equilParams(filename)
    
    Ip = ep.g['Ip']
    Bt0 = ep.g['Bt0']
    q95 = np.interp(0.95, ep.g['psiN'],ep.g['q'])
    Rc1 = 1.575; Zc1 = -1.30
    Rc2 = 1.72; Zc2 = -1.51
    alpha1 = pitch_angle(ep, Rc1, Zc1)
    alpha2 = pitch_angle(ep, Rc2, Zc2)

    return Bt0, Ip, q95, alpha1, alpha2
