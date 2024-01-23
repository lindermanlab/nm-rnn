import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
from jax.tree_util import tree_map
import optax
import numpy as np

# import pickle as pkl

# import matplotlib.pyplot as plt
# import pdb

# from flax import linen as nn
import functools
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from utils import *

# CODE FOR NM-LR-RNN

def nm_rnn(params, x0, z0, inputs, tau_x, tau_z, nln=jnp.tanh):
    """
    Arguments:
    - params
    - x0: initial state of LR-RNN
    - z0: initial state of NM network
    - inputs
    - tau_x : decay constant for LR-RNN
    - tau_z : decay constant for NM network
    """

    # U -- input dim, N -- hidden_dim, R -- rank, dim_nm -- hidden dim of NM network

    U = params["row_factors"]               # N x R
    V = params["column_factors"]            # N x R
    B_xu = params["input_weights"]          # N x U
    C = params["readout_weights"]           # O x N
    W_zz = params["nm_rec_weight"]          # dim_nm x dim_nm
    B_zu = params["nm_input_weight"]        # dim_nm x U
    W_kz = params["nm_sigmoid_weight"]      # R x dim_nm
    b_k = params["nm_sigmoid_bias"]         # R
    W_zx = params['nm_feedback_weight']     # dim_nm x N
    b_z = params['nm_feedback_bias']        # dim_nm

    # W_xz = params['input_gating_weight']    # N x dim_nm
    # b_x = params['input_gating_bias']       # N

    N = x0.shape[0]
    R = U.shape[1]

    def _step(x_and_z, u):
        x_p, z_p = x_and_z

        # update z
        z = (1.0 - (1. / tau_z)) * z_p
        z += (1. / tau_z) * W_zz @ nln(z_p)
        z += (1. / tau_z) * B_zu @ u
        z += (1. / tau_z) * (W_zx @ nln(x_p) + b_z)

        # update x
        s = jax.nn.sigmoid(W_kz @ z + b_k)
        x = (1.0 - (1. / tau_x)) * x_p
        h = V.T @ nln(x_p)
        x += (1. / (tau_x * N)) * (U * s) @ h # divide by N

        # x += (1. / tau_x) * nln(W_xz @ z + b_x)
        x += (1. / tau_x) * B_xu @ u

        # calculate y
        y = C @ x
        return (x, z), (y, x, z)

    _, (ys, xs, zs) = lax.scan(_step, (x0, z0), inputs)

    return ys, xs, zs

batched_nm_rnn = vmap(nm_rnn, in_axes=(None, None, None, 0, None, None))

def random_nmrnn_params(key, u, n, r, m, k, o, g=1.0):
    """Generate random low-rank RNN parameters

    Arguments:
    u:  number of inputs
    n:  number of neurons in main network
    r:  rank of main network
    m:  number of neurons in NM network
    k:  dimension of NM input affecting weight matrix (either 1 or r)
    o:  number of outputs
    """
    skeys = jr.split(key, 12)

    return {'row_factors' : jr.normal(skeys[0], (n,r)) / jnp.sqrt(r), #row_factors,
            'column_factors' : jr.normal(skeys[1], (n,r)) / jnp.sqrt(r), # column_factors,
            'input_weights' : jr.normal(skeys[2], (n,u)) / jnp.sqrt(u),
            'readout_weights' : jr.normal(skeys[3], (o,n)) / jnp.sqrt(n),
            'nm_rec_weight' : jr.normal(skeys[4], (m,m)) / jnp.sqrt(m),
            'nm_input_weight' : jr.normal(skeys[5], (m,u)) / jnp.sqrt(u),
            'nm_sigmoid_weight' : jr.normal(skeys[6], (k,m)) / jnp.sqrt(m),
            'nm_sigmoid_bias' : jr.normal(skeys[7], (k,)) / jnp.sqrt(k),
            'nm_feedback_weight' : jr.normal(skeys[8], (m, n)) / jnp.sqrt(n),
            'nm_feedback_bias' : jr.normal(skeys[9], (m,)) / jnp.sqrt(m)
            # 'input_gating_weight' : jr.normal(skeys[10], (n, m)) / jnp.sqrt(m),
            # 'input_gating_bias' : jr.normal(skeys[11], (n,)) / jnp.sqrt(n)
            }


def fit_element_finder_nm_lrrnn(params, optimizer, x0, z0, num_batches, batch_size, tau_x, tau_z, seq_len, key, loss_window_size=10):
    opt_state = optimizer.init(params)

    @jit
    def _loss(params, batch_inputs, batch_targets):
        ys, xs, zs = batched_nm_rnn(params, x0, z0, batch_inputs, tau_x, tau_z)
        return jnp.mean(((ys[:, -1, 0] - batch_targets)**2))

    @jit
    def _step(params, opt_state, batch_inputs, batch_targets):
        loss_value, grads = jax.value_and_grad(_loss)(params, batch_inputs, batch_targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    losses = []
    min_loss = None
    lowest_loss_params = None
    for n in range(num_batches):
        key, subkey = jr.split(key, 2)
        batch_inputs, batch_targets = generate_sequences_and_inds(batch_size, seq_len, subkey)


        params, opt_state, loss_value = _step(params, opt_state, batch_inputs, batch_targets)
        losses.append(loss_value)

        if min_loss is None or loss_value < min_loss:
            min_loss = loss_value
            lowest_loss_params = params

        if n % 500 == 0:
            print(f'step {n}, loss: {loss_value}')

            # sample_input, sample_target = batch_inputs[0], batch_targets[0]
            # # print(sample_input.shape)
            # sample_ys, sample_xs, sample_zs = nm_rnn(params, x0, z0, sample_input, tau_x, tau_z)
            # print(sample_zs)

    window_avgd_loss = 0
    for j in range(1, loss_window_size+1):
        window_avgd_loss += losses[-j]
    window_avgd_loss = window_avgd_loss / loss_window_size

    return params, lowest_loss_params, losses, min_loss, window_avgd_loss

if __name__ == "__main__":
    seq_len = 25
    batch_size = 128
    num_batches = 20000
    key = jr.PRNGKey(13)

    optimizer = optax.adam(0.01, 0.9, 0.999, 1e-07)

    # good runset: (M, N, R) = (5, 18, 8)

    # Initialization parameters
    U = 1  # input dim
    N = 18  # hidden state dim
    R = 8  # rank of RNN
    M = 5  # dim nm
    K = R  # rank R (factor specific) or 1 (global)
    O = 1  # output dimension
    tau_z = 10.
    tau_x = 2.

    key, skey, sskey = jr.split(key, 3)
    x0 = jr.normal(skey, (N,)) / jnp.sqrt(N)
    z0 = jr.normal(sskey, (M,)) / jnp.sqrt(M)

    params = random_nmrnn_params(key, U, N, R, M, K, O)

    params, lowest_loss_params, losses, min_loss, window_avgd_loss = fit_element_finder_nm_lrrnn(params, optimizer, x0,
                                                                                                 z0, num_batches,
                                                                                                 batch_size, tau_x,
                                                                                                 tau_z, seq_len, key)
