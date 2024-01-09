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

# CODE FOR LOW RANK RNN

def low_rank_rnn(params, x0, inputs, tau, nln=jnp.tanh):
    """
    Arguments:
    - params
    - x0
    - inputs
    - tau   : decay constant
    """

    U = params["row_factors"]       # D x R
    V = params["column_factors"]    # D x R
    B = params["input_weights"]     # D x M
    C = params["readout_weights"]   # O x D

    N = x0.shape[0]

    def _step(x, u):
        # x = (1.0 - (1. / tau)) * x
        # x += 1. / tau * U @ V.T @ nln(x)
        x = (1.0 - (1. / tau)) * x + (1. / (tau * N)) * U @ V.T @ nln(x) # divide by N
        # changed the above bc nln was being computed on (1-1/tau)x
        x += (1. / tau) * B @ u
        y = C @ x
        return x, (y, x)

    _, (ys, xs) = lax.scan(_step, x0, inputs)

    return ys, xs

batched_low_rank_rnn = vmap(low_rank_rnn, in_axes=(None, None, 0, None))

def batched_loss(params, x0, batch_inputs, tau, batch_targets):
    ys, _ = batched_low_rank_rnn(params, x0, batch_inputs, tau)
    return jnp.mean(((ys - batch_targets)**2))

def fit_element_finder_lrrnn(params, optimizer, x0, num_batches, batch_size, tau, seq_len, key):
    opt_state = optimizer.init(params)

    @jit
    def _loss(params, x0, batch_inputs, tau, batch_targets):
        ys, _ = batched_low_rank_rnn(params, x0, batch_inputs, tau)
        return jnp.mean(((ys[:, -1, 0] - batch_targets)**2))

    @jit
    def _step(params, opt_state, x0, batch_inputs, batch_targets):
        loss_value, grads = jax.value_and_grad(_loss)(params, x0, batch_inputs, tau, batch_targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    for n in range(num_batches):
        key, subkey = jr.split(key, 2)
        batch_inputs, batch_targets = generate_sequences_and_inds(batch_size, seq_len, subkey)


        params, opt_state, loss_value = _step(params, opt_state, x0, batch_inputs, batch_targets)
        losses.append(loss_value)
        if n % 500 == 0:
            print(f'step {n}, loss: {loss_value}')

    return params, losses


def random_lrrnn_params(key, u, n, r, o, g=1.0):
  """Generate random low-rank RNN parameters"""

  skeys = jr.split(key, 4)
#   hscale = 0.1
  ifactor = 1.0 / jnp.sqrt(u) # scaling of input weights
  hfactor = g / jnp.sqrt(n) # scaling of recurrent weights
  pfactor = 1.0 / jnp.sqrt(n) # scaling of output weights
  return {'row_factors' : jr.normal(skeys[0], (n,r)) * hfactor,
          'column_factors' : jr.normal(skeys[1], (n,r)) * hfactor,
          'input_weights' : jr.normal(skeys[2], (n,u)) *  ifactor,
          'readout_weights' : jr.normal(skeys[3], (o,n)) * pfactor}
