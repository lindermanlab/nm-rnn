# for Sherlock, make sure to use local copy of nmrnn
# import sys
# sys.path.insert(1, '/home/groups/swl1/nm-rnn')

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

# import matplotlib
# matplotlib.use('TkAgg') # need this on my machine for some reason
import matplotlib.pyplot as plt
import wandb

from nmrnn.data_generation import sample_memory_pro, sample_memory_anti, sample_delay_pro, sample_delay_anti, random_trials
from nmrnn.util import random_nmrnn_params, log_wandb_model
from nmrnn.fitting import fit_mwg_nm_rnn
from nmrnn.rnn_code import batched_nm_rnn

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    N = 100,    # hidden state dim
    R = 4,      # rank of RNN
    U = 7,      # input dim (3+num_tasks)
    O = 3,      # output dimension
    M = 5,      # NM dimension
    # got rid of K for now, set to R by default
    #K = 2,      # NM sigmoid dimension (must be 1 or R)
    # Model Hyperparameters
    tau_x = 10,
    tau_z = 100,
    # Timing (task) parameters
    dt = 10,#ms
    # Data Generation
    task_list = ['delay_pro', 'delay_anti', 'memory_pro', 'memory_anti'],
    T = 100,
    num_trials = 500,
    # Training
    num_full_train_iters = 100_000,
    keyind = 13,
)

# wandb stuff
projectname = "nm-rnn-multitask"
wandb.init(config=default_config, project=projectname, entity='nm-rnn')
config = wandb.config

# unpack tasks
task_list = []
if 'delay_pro' in config['task_list']:
    task_list.append(sample_delay_pro)
if 'delay_anti' in config['task_list']:
    task_list.append(sample_delay_anti)
if 'memory_pro' in config['task_list']:
    task_list.append(sample_memory_pro)
if 'memory_anti' in config['task_list']:
    task_list.append(sample_memory_anti)

# data generation
task_order, samples_in, samples_out = random_trials(
    jr.PRNGKey(config['keyind']), 
    task_list, 
    config['T'], 
    config['num_trials'])

key = jr.PRNGKey(config['keyind'])

# define a simple optimizer
# optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=1e-3),
)

x0 = jnp.ones((config['N'],))*0.1
z0 = jnp.ones((config['M'],))*0.1
masks = jnp.ones_like(samples_out)

# generate random initial parameters
init_params = random_nmrnn_params(key, config['U'], config['N'], config['R'],
                                  config['M'], config['R'], config['O'])

# train on all params
params, losses = fit_mwg_nm_rnn(samples_in.transpose((0,2,1)), samples_out.transpose((0,2,1)), masks.transpose((0,2,1)),
                                init_params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'], plots=False, wandb_log=True, final_wandb_plot=True)

# log model
log_wandb_model(params, "multitask_nmrnn_r{}_n{}_m{}".format(config['R'],config['N'],config['M']), 'model')

