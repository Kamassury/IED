# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:58:33 2021

@author: J Kamassury
"""
import numpy as np
from uteis import SBND, getfer, biawgn

def BER_FER(max_nfe, H, model, batch_size, T= 1, p_initial = 0, p_end = 100, interv = 0.5):
    ''' Estimate BER and FER as a function of Eb_N0_dB '''
    berr, ferr = [], []
    x = np.arange(p_initial, p_end + interv, interv)
    for b in range(np.size(x)):
        code = SBND(H, model, T=T, method=1, get_all=True, beta= 1)
        result = getfer(biawgn(code, EbN0_dB= x[b], batch_size = batch_size, random_state=1), max_nfe=max_nfe, verbose=False)
        berr.append(result['ser'])
        ferr.append(result['fer'])
    return berr, ferr

#pickle.dump(ber_45, open('ber_jcis_45_git.pkl', 'wb'))
#pickle.dump(fer_45, open('fer_jcis_45_git.pkl', 'wb'))