# -*- coding: utf-8 -*-
"""
@author: J. Kamassury

Neural network

"""
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import Input, BatchNormalization, Concatenate

def model_45(n=63, k=45, num_layers=6, hidden_units=300, activation='relu', kernel_initializer='normal'):
    '''Neural network for the BCH(63,45) code
    from Kavvousanos et al.: Hardware Implementation Aspects of a Syndrome-based Neural Network Decoder for BCH Codes
    '''
    input_length = n + n - k
    output_length = n 
    inputs = Input(shape=(input_length,))
    x = inputs
    for i in range(num_layers):
        x = Dense(hidden_units, kernel_initializer=kernel_initializer, activation=activation)(x)
    outputs = Dense(output_length, kernel_initializer=kernel_initializer, activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=outputs)
    return model

def model_36(n=63, k=36, num_layers=7, activation='relu', BN= False, kernel_initializer='normal'):
    '''Neural network for the BCH(63,36) code
    '''
    input_length = n + n - k
    output_length = n 
    inputs = Input(shape=(input_length,))
    x = inputs
    for i in range(num_layers):
        x = Dense(8*n, kernel_initializer=kernel_initializer, activation=activation)(x)
        if i==0:
            skip = x
        if BN:
            x = BatchNormalization()(x)
        if i==2:    
            x = Concatenate(axis=1)([x, skip])
    outputs = Dense(output_length, kernel_initializer=kernel_initializer, activation='sigmoid')(x)
    model = Model(inputs=inputs,outputs=outputs)
    return model