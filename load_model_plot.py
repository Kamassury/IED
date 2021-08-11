# -*- coding: utf-8 -*-
"""
@author: J. Kamassury

"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from uteis import bch_par
from ber_fer_result import BER_FER

'''
To load and perform the inference, uncomment the following commands
'''

# ---------------------------- Load Model --------------------------------
# Example: BCH(63, 45) code
#model = load_model('models/model_63_45.h5', compile=False)
n, k = 63,45
H = bch_par(n, k)

# ---------------------------- Inference ---------------------------------- 

p_in, interv, p_end =  2, 0.5, 6 #EbN0_dB initial, interval, EbN0_dB end

#max_nfe = 100 # Number of block errors
#batch_size, T = 1024, 5 # batch size and numbers of iterations

#ber_45, bler_45 = BER_FER(max_nfe, H, model, batch_size, T, 
#                 p_initial = p_initial, p_end = p_end, interv = interval)

#  -------------------------- Save the results ---------------------------- 
path_bler = 'results/BCH_63_45/bler_jcis_45.pkl'
path_ber =  'results/BCH_63_45/ber_jcis_45.pkl'


#pickle.dump(ber_45, open(path_ber, 'wb'))
#pickle.dump(bler_45, open(path_bler, 'wb'))


# -------------- Plot the results of performance metrics ------------------    
x = np.arange(p_in, p_end + interv, interv)
    
if n == 63 and k==45: #  ------------- bch(63,45) -------------------------
    # BLER
    BLER_45 = np.array(pickle.load(open(path_bler, 'rb'))) 
    HDD_BD_45 = np.array([0.6066387,0.46542759,0.3246810,0.202522,0.1111771, 0.05292534, 0.0215510,0.0074106,0.002125]) 
    Lugosch_45 = np.array([0.8118035, 0.65721175, 0.54762502, 0.37695373,0.2563149, 0.1373394, 0.07275, 0.028778, 0.01106693])
    Beery_45 = np.array([0, 0, 0, 0, 0.095576, 0.03813565, 0.0141111, 0.00398005, 0.00108923])
    ML_45 = np.array([100/691, 100/1360, 100/3853, 100/12630, 100/46869, 100/210475, 100/1873797, 63/10000000, 0])
 
    plt.plot(x, HDD_BD_45, 'k--', label='HDD-BD')
    plt.plot(x, Lugosch_45, '.-',label='Lugosch and Gross [11]')
    plt.plot(x[4:], Beery_45[4:], '.-',label='Be\'ery et al. [12]')
    plt.plot(x, BLER_45[:,0], '.-', label='SBND [15]')
    plt.plot(x, BLER_45[:,1], '.-', label='SBND [15] + IED ($T=2$)')
    plt.plot(x, BLER_45[:,2], '.-', label='SBND [15] + IED ($T=3$)')
    plt.plot(x, BLER_45[:,3], '.-', label='SBND [15] + IED ($T=4$)')
    plt.plot(x, BLER_45[:,4], '.-', label='SBND [15] + IED ($T=5$)')
    plt.plot(x[:8], ML_45[:8], 'k-', label='ML')
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.ylabel('Block Error Rate (BLER)')
    plt.xlabel(r'$E_{b}/N_{0}$ (dB)')
    plt.grid(True)
    plt.gca().yaxis.grid(True, which='minor') 
    plt.xlim([2, 6]);
    plt.ylim([1e-6, 1])
    plt.show()
    
    # BER
    
    BER_45 = np.array(pickle.load(open(path_ber, 'rb')))
    Beery_45_ber = np.array([0, 0, 0, 0, 0.005633261064922549, 0.002114598712892244, 0.0007718590381296847, 0.00019579126532222573, 0.0000503648633052628])
    ML_45 = np.array([0, 0, 0.017152947054760292, 0.007574436830992119, 0.003398775709818745, 0.0008703124054609424, 0.00023011713371470517, 0.00004010959343675761, 0.0000074540069128560486])

    plt.plot(x[4:], Beery_45_ber[4:], '.-', color='tab:orange', label='Be\'ery et al. [12]')
    plt.plot(x, BER_45[:,0], '.-', color='tab:green', label='SBND [15]')
    plt.plot(x, BER_45[:,1], '.-', color='tab:red', label='SBND [15] + IED ($T=2$)')
    plt.plot(x, BER_45[:,2], '.-', color='tab:purple', label='SBND [15] + IED ($T=3$)')
    plt.plot(x, BER_45[:,3], '.-', color='tab:brown', label='SBND [15] + IED ($T=4$)')
    plt.plot(x, BER_45[:,4], '.-', color='tab:pink', label='SBND [15] + IED ($T=5$)')
    plt.plot(x[:7], ML_45[2:], 'k-', label='ML (OSD) [18]')
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.ylabel('Bit Error Rate (BER)')
    plt.xlabel(r'$E_{b}/N_{0}$ (dB)')
    plt.grid(True)
    plt.gca().yaxis.grid(True, which='minor') 
    plt.xlim([2, 5.5]);
    plt.ylim([1e-6, 1e-1])
    plt.show()  
        
    
elif n==63 and k==36: 
    #BLER
    BLER_36 = np.array(pickle.load(open(path_bler, 'rb'))) # carregar
    HDD_BD_36 = np.array([0.496087284080289, 0.356667940600864,  0.230169495712008, 0.130835832617526, 0.064290648194155, 0.026808192077429, 0.009314946022484, 0.002649105381165, 0.000605739519020]) 
    Lugosch_36 = np.array([ 0.8333307660470268, 0.7171684190893622, 0.605203785162798, 0.4399774290254319,0.3181303949523359, 0.18474311494330173, 0.10593034491930438, 0.045102771993574396, 0.01980899658245014])
    Beery_36 = np.array([0, 0, 0, 0, 0.18004542089330006, 0.07877871201174601, 0.034198965121827164, 0.0107404625018988, 0.0031199534460219125])
    DNN_relu = np.array([0.56835938, 0.44042969, 0.30566406, 0.19905599,  0.11484375, 0.05485026, 0.02317116, 0.00946514, 0.00304493])
    DNN_sig = np.array([0.56884766, 0.44238281, 0.29541016, 0.18066406, 0.09570312, 0.04736328, 0.01806641, 0.00488281 , 0.00097656])
    ML_36 = np.array([100/2231, 100/6911, 100/25536, 100/101381, 100/704547, 100/4792807, 0,0,0])

    plt.plot(x, HDD_BD_36,'k--', label='HDD-BD')
    plt.plot(x, Lugosch_36, '.-',label='Lugosch and Gross [11]')
    plt.plot(x[4:], Beery_36[4:], '.-',label='Be\'ery et al. [12]')
    plt.plot(x, DNN_relu, '.-',label='SBND (ReLU, w/o BN)')
    plt.plot(x, DNN_sig, '.-',label='SBND (w/o BN)')
    plt.plot(x, BLER_36[:,0], '.-', label='SBND')
    plt.plot(x, BLER_36[:,1], '.-', label='SBND + IED ($T=2$)')
    plt.plot(x, BLER_36[:,2], '.-', label='SBND + IED ($T=3$)')
    plt.plot(x, BLER_36[:,3], '.-', label='SBND + IED ($T=4$)')
    plt.plot(x, BLER_36[:,4], '.-', label='SBND + IED ($T=5$)')
    plt.plot(x[:6], ML_36[:6], 'k-', label='ML')
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.ylabel('Block Error Rate (BLER)')
    plt.xlabel(r'$E_{b}/N_{0}$ (dB)')
    plt.grid(True)
    plt.gca().yaxis.grid(True, which='minor') 
    plt.xlim([2, 6]);
    plt.ylim([1e-6,1])
    plt.show()

    # BER 
    BER_36 = np.array(pickle.load(open(path_ber, 'rb'))) 
    Beery_36 = np.array([0, 0, 0, 0, 0.012185319190439286, 0.0048606603967484355, 0.0019388921310371438, 0.0006011321519010453, 0.00017051384854334297])
    ML_36 = np.array([0.001020224273461357, 0.00022275429519995563, 0.00003258693229275407, 0.000004313031288378539])
    
    plt.plot(x[4:], Beery_36[4:], '.-', color='tab:orange', label='Be\'ery et al. [12]')
    plt.plot(x, BER_36[:,0], '.-', color='tab:purple', label='SBND')
    plt.plot(x, BER_36[:,1], '.-', color='tab:brown', label='SBND + IED ($T=2$)')
    plt.plot(x, BER_36[:,2], '.-', color='tab:pink', label='SBND + IED ($T=3$)')
    plt.plot(x, BER_36[:,3], '.-', color='tab:gray', label='SBND + IED ($T=4$)')
    plt.plot(x, BER_36[:,4], '.-', color='tab:olive', label='SBND + IED ($T=5$)')
    plt.plot(x[2:6], ML_36, 'k-', label='ML (OSD) [18]')
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.ylabel('Bit Error Rate (BER)')
    plt.xlabel(r'$E_{b}/N_{0}$ (dB)')
    plt.grid(True)
    plt.gca().yaxis.grid(True, which='minor') 
    plt.xlim([2, 6]);
    plt.ylim([1e-6,1e-1])
    plt.show()