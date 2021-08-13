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
from inference_plots import inference_plot


# ---------------------------- Load Model --------------------------------
# Example: BCH(63, 45) code
model = load_model('models/model_63_45.h5', compile=False)
H = bch_par(63, 45)

# ---------------------------- Inference ---------------------------------- 

p_in, interv, p_end =  2, 0.5, 6 #EbN0_dB initial, interval, EbN0_dB end

max_nfe = 100 # Number of block errors
batch_size, T = 1024, 5 # batch size and numbers of iterations

ber_45, bler_45 = BER_FER(max_nfe, H, model, batch_size, T, 
                 p_initial = p_in, p_end = p_end, interv = interv)

#  -------------------------- Save the results ---------------------------- 
path_bler = 'results/BCH_63_45/bler_jcis_45.pkl'
path_ber =  'results/BCH_63_45/ber_jcis_45.pkl'
pickle.dump(ber_45, open(path_ber, 'wb'))
pickle.dump(bler_45, open(path_bler, 'wb'))
inference_plot(n=63, k=45, path_bler = path_bler, path_ber = path_ber)

