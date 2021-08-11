# ----------------------------------------------------------------------------
import pickle
from uteis import bch_par
from ber_fer_result import BER_FER
from inference_plots import inference_plot
import models_nets
from get_training_model import training_nn

from statsmodels.stats.proportion import proportion_confint
confint = lambda k,n: proportion_confint(k, n, method='beta')



# -------------------- Get the parity check matrix H ------------------------ 
''' n, k = 63, 45 or  n, k = 63, 36 '''
 
n, k = 63,45
H = bch_par(n, k)


# -------- Construction of the neural network for the BCH(n, k) code ---------
''' For BCH (63,45):
    model = models_nets.model_45(63,45)  

    For code BCH(63,63):
    model = models_nets.model_36(63,36)
    model = models_nets.model_36(63,36, activation='sigmoid')
    model = models_nets.model_36(63,36, activation='sigmoid', BN=True)
'''

model = models_nets.model_45(n,k) 
model.summary()

# --------------------- Neural network model training ----------------------
#training_nn(model=model, H=H, tec='cic') # Remove the # if you want to take the training


# ------------------------- SBND (Inference) -----------------------------
#p_in, interval, p_end =  2, 0.5, 6 #EbN0_dB initial, interval, EbN0_dB end
#max_nfe = 100 # Number of block errors
#batch_size, T = 1024, 5 # batch size and numbers of iterations

#ber_result, bler_result = BER_FER(max_nfe, H, model, batch_size, T, p_initial = p_in, 
#                         p_end = p_end, interv = interval)

#  -------------------------- Save the results ----------------------------
# Example 
#path_bler = 'results/BCH_63_45/bler_jcis_45.pkl'
#path_ber =  'results/BCH_63_45/ber_jcis_45.pkl'

#pickle.dump(ber_result, open(path_bler, 'wb'))
#pickle.dump(bler_result, open(path_ber, 'wb'))


# ------------------------ Plot the results -------------------------------
'''Plot the results via function inference_plots '''
#inference_plot(n=n, k=k, path_bler = path_bler, path_ber = path_ber)
