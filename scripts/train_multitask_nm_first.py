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

from nmrnn.data_generation import sample_memory_pro, sample_memory_anti, sample_delay_pro, sample_delay_anti, random_trials, one_of_each
from nmrnn.util import random_nmrnn_params, log_wandb_model
from nmrnn.fitting import fit_mwg_nm_rnn, fit_mwg_nm_only
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
    num_nm_only_iters = 10_000,
    keyind = 13,
    orth_u = True
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


# split parameters for now
nm_params = {k: init_params[k] for k in ('nm_rec_weight', 'nm_input_weight', 'nm_sigmoid_weight', 'nm_sigmoid_intercept')}
lr_params = {k: init_params[k] for k in ('row_factors', 'column_factors', 'input_weights', 'readout_weights')}

if config['num_nm_only_iters'] != 0:
# # train on nm params only for a bit
    params, nm_only_losses = fit_mwg_nm_only(samples_in.transpose((0,2,1)), samples_out.transpose((0,2,1)), masks.transpose((0,2,1)), nm_params,
                                    lr_params, optimizer, x0, z0, config['num_nm_only_iters'],
                                            config['tau_x'], config['tau_z'], 
                                            plots=False, wandb_log=True, final_wandb_plot=False, orth_u=config['orth_u'])
else: params = init_params

# train on all params
params, losses = fit_mwg_nm_rnn(samples_in.transpose((0,2,1)), samples_out.transpose((0,2,1)), masks.transpose((0,2,1)),
                                params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'], 
                                plots=False, wandb_log=True, final_wandb_plot=True, orth_u=config['orth_u'])

# log model
log_wandb_model(params, "multitask_nmrnn_r{}_n{}_m{}".format(config['R'],config['N'],config['M']), 'model')

# another plot
rank = config['R']
key = jr.PRNGKey(13)
T = 100
task_list, samples_in, samples_out = one_of_each(key, T)
task_labels = ['delay_pro', 'delay_anti', 'memory_pro', 'memory_anti']

x0 = jnp.ones((100,))*0.1
z0 = jnp.ones((5,))*0.1

ys, xs, zs = batched_nm_rnn(params, x0, z0, samples_in.transpose((0,2,1)), 10, 100, config['orth_u'])

m = params['nm_sigmoid_weight']
b = params['nm_sigmoid_intercept']

fig, axes = plt.subplots(rank, 1, figsize=[10,rank*2])

for r, ax in enumerate(axes):
    for i in range(4):
        ax.plot(jax.nn.sigmoid((zs @ m.T + b)[i, :, r]), label=task_labels[i])
        ax.legend(loc='lower left')
        ax.set_ylabel('NM response')
        ax.set_ylim(-0.1,1.1)
ax.set_xlabel('time')

wandb.log({'one_of_each':wandb.Image(fig)}, commit=True)