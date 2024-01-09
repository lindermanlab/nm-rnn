import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
from jax.tree_util import tree_map
import optax
import numpy as np

import pickle as pkl

import matplotlib.pyplot as plt
import pdb

from flax import linen as nn
import functools
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import *

def simple_lstm(params, c0, h0, inputs, gate_fn=jax.nn.sigmoid, activation_fn=jnp.tanh):
    """
    Arguments:
    - params: dict of LSTM weights and cell states
    - c0: initial cell (memory) state
    - h0: initial hidden state
    - inputs: sequence of inputs passed in, with batch dims first
    """
    # N -- hidden size, M -- input_size, O -- output_size

    # (c0, h0) = init_carry

    N = h0.shape[0] # hidden size of LSTM

    W_iu = params['weights_iu']     # N x M
    W_ih = params['weights_ih']     # N x N
    b_ih = params['bias_ih']        # scalar

    W_fu = params['weights_fu']     # N x M
    W_fh = params['weights_fh']     # N x N
    b_fh = params['bias_fh']        # scalar

    W_gu = params['weights_gu']     # N x M
    W_gh = params['weights_gh']     # N x N
    b_gh = params['bias_gh']        # scalar

    W_ou = params['weights_ou']     # N x M
    W_oh = params['weights_oh']     # N x N
    b_oh = params['bias_oh']        # scalar

    C = params["readout_weights"]   # O x N

    def _step(carry, u):
        c, h = carry
        i = gate_fn(W_iu @ u + W_ih @ h + b_ih)
        f = gate_fn(W_fu @ u + W_fh @ h + b_fh)
        g = activation_fn(W_gu @ u + W_gh @ h + b_gh)
        o = gate_fn(W_ou @ u + W_oh @ h + b_oh)

        new_c = f * c + i * g
        new_h = o * activation_fn(new_c)

        y = C @ new_h

        return (new_c, new_h), y

    if 'embedding_weights' in params:
      embedding = params['embedding_weights']
      inputs = embedding[inputs]    # make input batch_size x max_len x num_features in the case of text sentiment analysis

    carry, y = lax.scan(_step, (c0, h0), inputs)
    return carry, y

def initialize_carry(hidden_size, key):
    key1, key2 = jr.split(key)
    mem_shape = (hidden_size, )
    scale_factor = 1.0 / jnp.sqrt(hidden_size)

    c = jr.normal(key1, mem_shape) * scale_factor
    h = jr.normal(key2, mem_shape) * scale_factor
    carry = (c, h)
    return carry

# should initial carry also be batched?
batched_simple_lstm = vmap(simple_lstm, in_axes=(None, None, None, 0))

def lstm_batched_loss(params, c0, h0, batch_inputs, batch_targets):
    _, ys = batched_simple_lstm(params, c0, h0, batch_inputs)
    return jnp.mean(((ys - batch_targets)**2))

def random_lstm_params(key, u, n, o, embed_size=None, n_embeds=None):
    """Generate random low-rank RNN parameters"""

    skeys = jr.split(key, 14)
    lstm_inp = u
    if embed_size is not None:
        assert n_embeds is not None
        lstm_inp = embed_size

    ifactor = 1.0 / jnp.sqrt(lstm_inp) # scaling of input weights
    hfactor = 1.0 / jnp.sqrt(n) # scaling of recurrent weights
    pfactor = 1.0 / jnp.sqrt(n) # scaling of output weights
    params = {'weights_iu' : jr.normal(skeys[0], (n,lstm_inp)) * ifactor,
            'weights_ih' : jr.normal(skeys[1], (n,n)) * hfactor,
            'bias_ih'  : jr.normal(skeys[2], (n,)) * hfactor,
            'weights_fu' : jr.normal(skeys[3], (n,lstm_inp)) * ifactor,
            'weights_fh' : jr.normal(skeys[4], (n,n)) * hfactor,
            'bias_fh'  : jr.normal(skeys[5], (n,)) * hfactor,
            'weights_gu' : jr.normal(skeys[6], (n,lstm_inp)) * ifactor,
            'weights_gh' : jr.normal(skeys[7], (n,n)) * hfactor,
            'bias_gh'  : jr.normal(skeys[8], (n,)) * hfactor,
            'weights_ou' : jr.normal(skeys[9], (n,lstm_inp)) * ifactor,
            'weights_oh' : jr.normal(skeys[10], (n,n)) * hfactor,
            'bias_oh'  : jr.normal(skeys[11], (n,)) * hfactor,
            'readout_weights' : jr.normal(skeys[12], (o,n)) * pfactor}
    if embed_size is not None:
        params['embedding_weights'] = jr.normal(skeys[13], (n_embeds, embed_size))

    return params

def fit_element_finder_lstm(params, optimizer, init_carry, num_batches, batch_size, seq_len, key):
    opt_state = optimizer.init(params)

    @jit
    def _loss(params, c0, h0, batch_inputs, batch_targets):
        _, ys = batched_simple_lstm(params, c0, h0, batch_inputs)
        return jnp.mean(((ys[:, -1, 0] - batch_targets)**2))

    @jit
    def _step(params, opt_state, init_carry, batch_inputs, batch_targets):
        (c0, h0) = init_carry
        loss_value, grads = jax.value_and_grad(_loss)(params, c0, h0, batch_inputs, batch_targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    for n in range(num_batches):
        key, subkey = jr.split(key, 2)
        batch_inputs, batch_targets = generate_sequences_and_inds(batch_size, seq_len, subkey)

        params, opt_state, loss_value = _step(params, opt_state, init_carry, batch_inputs, batch_targets)
        losses.append(loss_value)
        if n % 500 == 0:
            print(f'step {n}, loss: {loss_value}')

    return params, losses

