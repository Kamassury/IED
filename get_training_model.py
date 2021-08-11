
"""
@author: J. Kamassury

"""
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from uteis import training_generator, PrintFER
import tensorflow_addons as tfa

def training_nn(model, H, loss='binary_crossentropy', lr = 1e-3, 
                batch_size=2048, spe=100, epochs=10000, EbN0_dB=4, tec='red'): 
    filepath="weights.best.hdf5"

    if tec ==  'red':
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        patience = 10
        model.compile(loss=loss, optimizer=Adam(lr=lr))
        model.fit(training_generator(H, EbN0_dB, batch_size, random_state=0), steps_per_epoch=spe, epochs=epochs,
                  callbacks=[PrintFER(H, EbN0_dB, T=None, random_state=1, max_nfe=100), 
                  ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience), ModelCheckpoint('callbacks_list')])
    elif tec=='cic':
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-5, maximal_learning_rate=lr,
            scale_fn=lambda x: 1/(2.**(x-1)), step_size=64)
        model.compile(loss=loss, optimizer=Adam(clr))
        model.fit(training_generator(H, EbN0_dB, batch_size, random_state=0), steps_per_epoch=150, epochs=epochs,
                  callbacks=[PrintFER(H, EbN0_dB, T=None, random_state=1, max_nfe=100),
        ModelCheckpoint('callbacks_list')])