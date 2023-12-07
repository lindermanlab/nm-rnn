import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax

import matplotlib.pyplot as plt
import wandb

from nmrnn.data_generation import sample_all
from nmrnn.util import random_nmrnn_params, log_wandb_model
from nmrnn.fitting import fit_mwg_nm_rnn, fit_mwg_nm_only

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    N = 100,    # hidden state dim
    R = 3,      # rank of RNN
    U = 3,      # input dim
    O = 1,      # output dimension
    M = 5,      # NM dimension
    K = 3,      # NM sigmoid dimension
    # Model Hyperparameters
    tau_x = 10,
    tau_z = 100,
    # Timing (task) parameters
    dt = 10,#ms
    # Data Generation
    T = 110,
    measure_min = 10,
    measure_max = 20,
    intervals = [[12, 14, 16, 18]],
    delay = 15,
    # Training
    num_nm_only_iters = 10_000,
    num_full_train_iters = 10_000,
    keyind = 13,
)

projectname = "nm-rnn-mwg"
wandb.init(config=default_config, project=projectname)
config = wandb.config

all_inputs, all_outputs, all_masks = sample_all(config['T'],
           jnp.array(config['intervals']),
            config['measure_min'],
            config['measure_max'],
            config['delay'],)

key = jr.PRNGKey(config['keyind'])

# define a simple optimizer
# optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=1e-2),
)

x0 = jnp.ones((config['N'],))*0.1
z0 = jnp.ones((config['M'],))*0.1

init_params = random_nmrnn_params(key, config['U'], config['N'], config['R'],
                                  config['M'], config['K'], config['O'])

# split parameters for now (only train on nm params to start)
nm_params = {k: init_params[k] for k in ('readout_weights', 'nm_rec_weight', 'nm_input_weight', 'nm_sigmoid_weight', 'nm_sigmoid_intercept')}
other_params = {k: init_params[k] for k in ('row_factors', 'column_factors', 'input_weights')}

params, nm_only_losses = fit_mwg_nm_only(all_inputs, all_outputs, all_masks, nm_params,
                                 other_params, optimizer, x0, z0, config['num_nm_only_iters'],
                                         config['tau_x'], config['tau_z'], plots=False, wandb_log=True, final_wandb_plot=False)

params, losses = fit_mwg_nm_rnn(all_inputs, all_outputs, all_masks,
                                params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'], plots=False, wandb_log=True, final_wandb_plot=True)

log_wandb_model(params, "nmrnn_r{}_{}_{}".format(config['R'],config['N'],config['M']), 'model')