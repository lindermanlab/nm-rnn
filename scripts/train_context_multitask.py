# this script trains nm-rnn where the nm network is the only one to receive context

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

from nmrnn.data_generation import sample_memory_pro, sample_memory_anti, sample_delay_pro, sample_delay_anti, sample_dm1, sample_dm2, random_trials, one_of_each
from nmrnn.util import random_nmrnn_params, log_wandb_model, percent_correct
from nmrnn.fitting import fit_mwg_context_nm_rnn
from nmrnn.rnn_code import batched_context_nm_rnn, context_nm_rnn

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    N = 100,    # hidden state dim
    R = 4,      # rank of RNN
    U = 7,      # input dim (3+num_tasks)
    O = 2,      # output dimension
    M = 5,      # NM dimension
    # got rid of K for now, set to R by default
    #K = 2,      # NM sigmoid dimension (must be 1 or R)
    # Model Hyperparameters
    tau_x = 10,
    tau_z = 100,
    # Timing (task) parameters
    dt = 10,#ms
    # Data Generation
    delay_pro = True,
    delay_anti = True,
    memory_pro = True,
    memory_anti = True,
    dm_1 = False,
    dm_2 = False,
    T = 100,
    num_trials = 500,
    # Training
    num_full_train_iters = 100_000,
    keyind = 13,
    orth_u = True,
    fix_output=True
)

# wandb stuff
projectname = "nm-rnn-multitask"
wandb.init(config=default_config, project=projectname, entity='nm-rnn')
config = wandb.config

# unpack tasks
task_list = []
if config['delay_pro']:
    task_list.append(sample_delay_pro)
if config['delay_anti']:
    task_list.append(sample_delay_anti)
if config['memory_pro']:
    task_list.append(sample_memory_pro)
if config['memory_anti']:
    task_list.append(sample_memory_anti)
if config['dm_1']:
    task_list.append(sample_dm1)
if config['dm_2']:
    task_list.append(sample_dm2)

# data generation
task_order, samples_in, samples_out = random_trials(
    jr.PRNGKey(config['keyind']), 
    task_list, 
    config['T'], 
    config['num_trials'],
    config['fix_output'])

# separate out task and context inputs
task_samples_in = samples_in[:,:config['U']-len(task_list), :]
context_samples_in = samples_in[:,config['U']-len(task_list):, :]

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
# crop input weights since LR component won't get context
init_params['input_weights'] = init_params['input_weights'][:,:config['U']-len(task_list)]

# train on all params
params, losses = fit_mwg_context_nm_rnn(task_samples_in.transpose((0,2,1)), context_samples_in.transpose((0,2,1)), samples_out.transpose((0,2,1)), masks.transpose((0,2,1)),
                                init_params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'], 
                                plots=False, wandb_log=True, final_wandb_plot=True, orth_u=config['orth_u'])

# log model
log_wandb_model(params, "multitask_context_nmrnn_r{}_n{}_m{}".format(config['R'],config['N'],config['M']), 'model')

# log % correct
# TODO: currently only works when training on set of 4 tasks
if len(task_list) == 4:
    samples_in_test = jnp.load('test_inputs.npy')
    samples_out_test = jnp.load('test_outputs.npy')
    x0 = jnp.ones((100,))*0.1
    z0 = jnp.ones((5,))*0.1

    task_samples_in_test = samples_in_test[:,:-len(task_list), :]
    context_samples_in_test = samples_in_test[:,-len(task_list):, :]

    ys_test, _, _ = batched_context_nm_rnn(params, x0, z0, task_samples_in_test.transpose((0,2,1)), context_samples_in_test.transpose((0,2,1)), config['tau_x'], config['tau_z'], config['orth_u'])

    wandb.log({'percent_correct':percent_correct(samples_in_test, samples_out_test, ys_test)})

# example outputs plot
sample_task_inputs, sample_context_inputs, sample_targets, sample_masks = task_samples_in.transpose((0,2,1))[0], context_samples_in.transpose((0,2,1))[0], samples_out.transpose((0,2,1))[0], masks.transpose((0,2,1))[0] # grab a single trial to plot output

ys, _, zs = context_nm_rnn(params, x0, z0, sample_task_inputs, sample_context_inputs, config['tau_x'], config['tau_z'], orth_u=config['orth_u'])

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=[10, 6])
# ax0.xlabel('Timestep')
ax0.plot(sample_targets, label='True target')
ax0.plot(ys, label='RNN target')
ax0.legend()
ax1.set_xlabel('Timestep')
m = params['nm_sigmoid_weight']
b = params['nm_sigmoid_intercept']
ax1.plot(jax.nn.sigmoid((zs @ m.T + b)))
# ax1.legend()
wandb.log({'final_curves':wandb.Image(fig)}, commit=True)

# another plot
rank = config['R']
key = jr.PRNGKey(13)
T = 100
task_list_inds, samples_in, samples_out = one_of_each(key, task_list, T, fix_output=config['fix_output'])
task_labels = [task.__name__[7:] for task in task_list]

x0 = jnp.ones((100,))*0.1
z0 = jnp.ones((5,))*0.1

task_samples_in = samples_in[:,:-len(task_list), :]
context_samples_in = samples_in[:,-len(task_list):, :]

ys, xs, zs = batched_context_nm_rnn(params, x0, z0, task_samples_in.transpose((0,2,1)), context_samples_in.transpose((0,2,1)), config['tau_x'], config['tau_z'], config['orth_u'])

fig, axes = plt.subplots(rank, 1, figsize=[10,rank*2])

for r, ax in enumerate(axes):
    for i in range(4):
        ax.plot(jax.nn.sigmoid((zs @ m.T + b)[i, :, r]), label=task_labels[i])
        ax.legend(loc='lower left')
        ax.set_ylabel('NM response')
        ax.set_ylim(-0.1,1.1)
ax.set_xlabel('time')

wandb.log({'one_of_each':wandb.Image(fig)}, commit=True)