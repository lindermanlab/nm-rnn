import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax

import matplotlib.pyplot as plt

from nmrnn.data_generation import sample_all
from nmrnn.util import random_nmrnn_params
from nmrnn.fitting import fit_mwg_nm_rnn, fit_mwg_nm_only

# Data Generation
T = 110
measure_min = 10
measure_max = 20
# intervals = jnp.array([[8,  10, 12, 14],
#                        [16,  20, 24, 28]])
# intervals = jnp.array([[8,  10, 12, 14, 16, 20, 24, 28]])

intervals = jnp.array([[12, 14, 16, 18]])
delay = 15
# mask_pad = 3

all_inputs, all_outputs, all_masks = sample_all(T,
           intervals,
            measure_min,
            measure_max,
            delay,)

num_nm_only_iters = 10_000
num_full_train_iters = 50_000
keyind = 13
key = jr.PRNGKey(keyind)

# define a simple optimizer
# optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=1e-2),
)

# Initialization parameters
N = 100  # hidden state dim
R = 3   # rank of RNN
U = 3   # input dim
O = 1   # output dimension
M = 5   # NM dimension
K = 3   # NM sigmoid dimension

# Hyperparameters
tau_x = 10
x0 = jnp.ones((N,))*0.1
tau_z = 100
z0 = jnp.ones((M,))*0.1

# Timing (task) parameters
dt = 10#ms

init_params = random_nmrnn_params(key, U, N, R, M, K, O)

# split parameters for now (only train on nm params to start)
nm_params = {k: init_params[k] for k in ('readout_weights', 'nm_rec_weight', 'nm_input_weight', 'nm_sigmoid_weight', 'nm_sigmoid_intercept')}
other_params = {k: init_params[k] for k in ('row_factors', 'column_factors', 'input_weights')}

params, nm_only_losses = fit_mwg_nm_only(all_inputs, all_outputs, all_masks, nm_params,
                                 other_params, optimizer, x0, z0, num_nm_only_iters,
                                         tau_x, tau_z, plots=False)

params, losses = fit_mwg_nm_rnn(all_inputs, all_outputs, all_masks,
                                params, optimizer, x0, z0, num_full_train_iters,
                                tau_x, tau_z, plots=False)