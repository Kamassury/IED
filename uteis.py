# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:59:27 2021

@author: kysnney kamassury

"""
import numpy as np
import keras 

def bch_par(n, k):
    '''Parity-check matrix of BCH(n,k) code'''
    genpoly = {
        # Assumes g[i] == g_{n-k-i} (MATLAB: g = bchgenpoly(n,k))
        (63, 45): [1,1,1,1,0,0,0,0,0,1,0,1,1,0,0,1,1,1,1],
        (63, 36): [1,0,0,0,0,1,1,0,1,1,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1]
    }
    g = genpoly[(n,k)]
    assert len(g) == n-k+1
    assert g[0] == 1
    g = np.array(g, dtype=int)
    G = np.zeros((k,n), dtype=int)
    for j in range(k):
        G[j,j:j+len(g)] = g
        for i in range(j-1,-1,-1):
            if G[i,j] == 1:
                G[i] = np.mod(G[i] + G[j], 2)
    assert np.all(G[:,0:k] == np.eye(k))
    H = np.c_[G[:,k:].T, np.eye(n-k, dtype=int)]
    return H


def training_generator(H, EbN0_dB, batch_size, random_state=None):
    rng = np.random.RandomState(random_state)
    n = H.shape[1]
    k = n - H.shape[0]
    R = k/n
    EbN0 = 10**(EbN0_dB/10)
    sigma = 1/np.sqrt(2*R*EbN0)
    c = np.zeros((batch_size, n), dtype=int)
    x = 1-2*c
    while True:
        z = sigma*rng.randn(*x.shape)
        y = x + z
        yb = (y < 0).astype(int)
        e = np.mod(yb + c, 2)
        s = np.mod(yb@H.T, 2).astype(int)
        a = np.abs(y)
        yield np.c_[s,a], e
        
def getfer(error_generator, max_nfe=100, min_fer=1e-6, verbose=True):
    '''Example:
    code = SBND(H, model)
    result = getfer(biawgn(code, EbN0_dB=4, random_state=1), max_nfe=100)
    '''
    from tqdm import tqdm
    if type(verbose).__name__ == 'tqdm':
        pbar = verbose
        verbose = True
    elif verbose:
        with tqdm(total=max_nfe) as pbar:
            result = getfer(error_generator, max_nfe, min_fer, pbar)
        return result
    nsymbols = 0
    nframes = 0
    stop_below = False
    while True:
        err = next(error_generator)
        if nframes == 0:
            nse = np.zeros(err.shape[:-2], dtype=int) + 0
            nfe = np.zeros(err.shape[:-2], dtype=int) + 0
        nse += err.sum(axis=-1).sum(axis=-1)
        nfe_new = err.any(axis=-1).sum(axis=-1)
        nfe += nfe_new #err.any(axis=-1).sum(axis=-1)
        nsymbols += err.shape[-2]*err.shape[-1]
        nframes += err.shape[-2]
        ser = nse/nsymbols
        fer = nfe/nframes
        
        if verbose:
            pbar.update(np.min(nfe_new))
            pbar.set_description('fer = %e' % (np.min(fer)))
            pbar.set_postfix(nframes=nframes, ser=np.min(ser))

        if np.min(nfe) >= max_nfe:
            break
        ferbound = max_nfe/(nframes + max_nfe - np.min(nfe))
        if ferbound < min_fer:
            stop_below = True
            break
    result = {'fer':fer, 'nfe':nfe, 'nframes':nframes, 'ser':ser, 'nse':nse, 'nsymbols':nsymbols, 'stop_below':stop_below}
    return result

def biawgn(code, EbN0_dB, batch_size=2048, random_state=None):
    '''Example:
    code = SBND(H, model)
    result = getfer(biawgn(code, EbN0_dB=4, random_state=1), max_nfe=100)
    '''
    # Requires code.k, code.n, code.decode()
    rng = np.random.RandomState(random_state)
    R = code.k/code.n
    EbN0 = 10**(EbN0_dB/10)
    sigma = 1/np.sqrt(2*R*EbN0)
    while True:
        # zero codeword assumption
        c = np.zeros((batch_size, code.n), dtype=int)
        x = 1-2*c
        z = sigma*rng.randn(*x.shape)
        y = x + z
        err = (code.decode(y) != c)
        yield err

def getniter(code, error_generator, nframes):   
    '''USE ONLY TO MEASURE NUMBER OF ITERATIONS
    Example:
    code = SBND(H, model)
    result = getniter(code, biawgn(code, EbN0_dB=4, random_state=1), nframes=10000)
    '''
    err = next(error_generator)
    assert len(err.shape) == 2
    batch_size, n = err.shape
    n_full_batches, partial_batch_size = np.divmod(nframes, batch_size)
    has_partial_batch = int(partial_batch_size > 0)
    nse = 0
    nfe = 0
    nsymbols = 0
    nframes = 0
    niter = 0
    for i in range(n_full_batches + has_partial_batch):
        if i > 0:
            err = next(error_generator)
        if i == n_full_batches:
            err = err[:partial_batch_size]
        nse += err.sum(axis=-1).sum(axis=-1)
        nfe += err.any(axis=-1).sum(axis=-1)
        nsymbols += np.prod(err.shape)
        nframes += err.shape[0]
        if hasattr(code, 'niter'):
            niter += code.niter[:err.shape[0]].sum()
    ser = nse/nsymbols
    fer = nfe/nframes
    result = {'niter':niter, 'fer':fer, 'nfe':nfe, 'nframes':nframes, 'ser':ser, 'nse':nse, 'nsymbols':nsymbols}
    return result

class SBND():
    '''Syndrome-Based Neural Decoder with Iterative Error Decimation'''
    def __init__(self, H, model, T=None, method=1, get_all=True, beta=0.5, old_beta=False):
        self.H = H
        self.n = H.shape[1]
        self.k = self.n - H.shape[0]
        self.model = model
        self.T = T
        self.method = method
        self.get_all = get_all
        self.beta = beta
        self.old_beta = old_beta
        
    def decode(self, y):
        if self.T is None:
            c_hat = self._decode(y)
        elif self.method == 1:
            c_hat = self._decode_ied_alt(y)
        if not self.get_all and len(c_hat.shape)>2:
            c_hat = c_hat[-1]
        return c_hat
        
    def _decode(self, y):
        yb = (y < 0).astype(int)
        s = np.mod(yb@self.H.T, 2).astype(int)
        a = np.abs(y)
        p = self.model.predict(np.c_[s,a])
        e_hat = (p > 0.5).astype(int)
        c_hat = np.mod(yb + e_hat, 2)
        return c_hat
    
    def _decode_ied_alt(self, y):
        assert self.T >= 0
        yb = (y < 0).astype(int)
        s = np.mod(yb@self.H.T, 2).astype(int)
        a = np.abs(y)
        p = np.zeros(y.shape)
        e_hat = np.zeros(y.shape)
        s_hat = np.zeros(s.shape)
        c_hat = np.zeros((self.T,*y.shape))
        self.niter = np.zeros(y.shape[0], dtype=int)
        for t in range(self.T):
            wrong = ~np.all(s_hat == s, axis=1)
            self.niter += wrong
            if np.all(~wrong):
                c_hat[t:] = np.mod(yb + e_hat, 2)
                break
            if t > 0:
                i = np.argmax(p[wrong,:], axis=1)
                yb[wrong,i] = 1-yb[wrong,i]
                s[wrong,:] = np.mod(s[wrong,:] + self.H.T[i,:], 2).astype(int)
                if not self.old_beta:
                    a[wrong,i] = self.beta*np.sign(a[wrong,i])
                else:
                    a[wrong,i] = self.beta*a[wrong,i]
            p_wrong = self.model.predict(np.c_[s[wrong,:],a[wrong,:]])
            p[wrong,:] = p_wrong
            e_hat[wrong,:] = (p[wrong,:] > 0.5).astype(int)
            s_hat[wrong,:] = np.mod(e_hat[wrong,:]@self.H.T, 2).astype(int)
            c_hat[t] = np.mod(yb + e_hat, 2)
        return c_hat

def soft_synd(y, H):
    import tensorflow as tf
    L = []
    for i in range(H.shape[0]):
        s = tf.boolean_mask(y, H[i,:]==1, axis=1)
        P = tf.reduce_prod(tf.sign(s),axis=1,keepdims=True)
        M = tf.reduce_min(tf.abs(s),axis=1,keepdims=True)
        L.append(P*M)
    synd = tf.concat(L,1)
    return synd

def syndrome_loss(target, output):
    # Assumes H is global
    import tensorflow as tf
    target = 1-2*target
    output = 1-2*output
    x = target*output
    loss = tf.reduce_mean(tf.maximum(1 - soft_synd(x, H), 0), axis=-1, keepdims=True)
    return loss

def custom_loss(target, output):
    '''Mix of binary cross-entropy loss and syndrome loss'''
    # Assumes synd_weight is global
    from tensorflow.keras.losses import binary_crossentropy as bce
    loss = (1-synd_weight)*bce(target, output) + (synd_weight)*syndrome_loss(target, output)
    return loss

class PrintFER(keras.callbacks.Callback):
    def __init__(self, H, EbN0_dB, T=None, batch_size=2048, random_state=1, max_nfe=100):
        self.H = H
        self.EbN0_dB = EbN0_dB
        self.T = T
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_nfe = max_nfe
    def on_epoch_end(self, epoch, logs=None):
        code = SBND(self.H, self.model, T=self.T, get_all=True)
        result = getfer(biawgn(code, self.EbN0_dB, self.batch_size, self.random_state), self.max_nfe, verbose=False)
        ser = result['ser']
        fer = result['fer']
        logs['ber'] = ser
        logs['fer'] = fer
        if self.T is None or self.T <= 1:
            print(' '*57 + 'ber: %g - fer: %g' % (ser, fer))
        else:
            print(' '*57 + 'ber: %g - fer: %g - fer-ied: %g' % (ser[0], fer[0], fer[-1]))