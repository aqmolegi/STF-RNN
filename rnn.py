import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np
from collections import OrderedDict
import os

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='IPs'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates

class RNNPredictor(object):

    __slots__ = ['VCount', 'Vt', 'window', 'd', 'dt', 'dh', 'IPs_values', 'TE_values', 'U_values', 'UT_values', 'V_values', 'W_values', 'h0_values',
                 'b_values', 'bh_values', 'IPs', 'TE', 'U', 'UT', 'W', 'V', 'b', 'bh', 'h0', 'params', 'names', 'predict', 'train', 'normalize']

    def __init__(self):
        
        self.VCount = v
        self.Vt = vt
        self.dt = dt
        self.window = window
        self.d = d
        self.dh = dh
            
        self.IPs_values = np.asarray((np.random.rand(self.VCount, d) - 0.5) / d, dtype=theano.config.floatX)
        self.TE_values = np.asarray((np.random.rand(vt, dt) - 0.5) / dt, dtype=theano.config.floatX)                        
        self.U_values = np.asarray((np.random.rand(d, dh) - 0.5) / dh, dtype=theano.config.floatX)            
        self.UT_values = np.asarray((np.random.rand(dt, dh) - 0.5) / dh, dtype=theano.config.floatX)
        self.W_values = np.asarray((np.random.rand(dh, dh) - 0.5) / dh, dtype=theano.config.floatX)
        self.V_values = np.asarray((np.random.rand(dh, self.VCount) - 0.5) / self.VCount, dtype=theano.config.floatX)
        self.h0_values = np.zeros((dh, ), dtype=theano.config.floatX)
        self.b_values   = np.zeros((v, ), dtype=theano.config.floatX)
        self.bh_values   = np.zeros((dh, ), dtype=theano.config.floatX)

        self.IPs = theano.shared(value = self.IPs_values, name='IPs', borrow=True)        
        self.U = theano.shared(value = self.U_values, name='U', borrow=True)
        self.TE = theano.shared(value = self.TE_values, name='TE', borrow=True)        
        self.UT = theano.shared(value = self.UT_values, name='UT', borrow=True)        
        self.W = theano.shared(value = self.W_values, name='W', borrow=True)
        self.V = theano.shared(value = self.V_values, name='V', borrow=True)
        self.h0  = theano.shared(value = self.h0_values, name='h0', borrow=True)
        self.b  = theano.shared(value = self.b_values, name='b', borrow=True)
        self.bh  = theano.shared(value = self.bh_values, name='bh', borrow=True)

        idxs = T.ivector('idxs')
        txs = T.ivector('txs')
        y = T.iscalar('y')
        x = self.IPs[idxs]
        t = self.TE[txs]
       
        def recurrence(X_t, T_t, h_tm1):
            h_t = Tanh(T.dot(X_t, self.U) + T.dot(T_t, self.UT) + T.dot(h_tm1, self.W) + self.bh)
            y_t = T.nnet.softmax(T.dot(h_t, self.V) + self.b)
            return h_t, y_t

        self.params = [ self.IPs, self.TE, self.U, self.UT, self.W, self.V, self.h0, self.b, self.bh ]
        self.names  = ['IPs', 'TE', 'U', 'UT', 'W', 'V', 'h0', 'b', 'bh']
      
        [h, s] , _ = theano.scan(fn=recurrence, sequences=[x, t], outputs_info=[self.h0, None], n_steps=x.shape[0])
        p_y_given_x_lastpos = s[-1,0,:]

        lr = 1e-3
        cost = -T.log(p_y_given_x_lastpos)[y]

        updates = sgd_updates_adadelta(self.params, cost)
      
        self.predict = theano.function(inputs = [idxs, txs], outputs = p_y_given_x_lastpos.argsort()[-3:][::-1])
        self.train = theano.function(inputs = [idxs, txs, y], outputs=cost, updates = updates)
