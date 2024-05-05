# this script trains nm-rnn where the nm network is the only one to receive context

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import matplotlib.pyplot as plt
import wandb

from nmrnn.data_generation import sample_memory_pro, sample_memory_anti, sample_delay_pro, sample_delay_anti, sample_dm1, sample_dm2, random_trials, one_of_each
from nmrnn.util import random_nmrnn_params, log_wandb_model, percent_correct
from nmrnn.fitting import fit_mwg_context_nm_rnn, fit_context_nm_only
from nmrnn.rnn_code import batched_context_nm_rnn, context_nm_rnn

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    N = 100,    # hidden state dim
    R = 4,      # rank of RNN
    U = 6,      # input dim (3+num_tasks)
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
    delay_pro = True,
    delay_anti = True,
    memory_pro = True,
    memory_anti = False,
    dm_1 = False,
    dm_2 = False,
    T = 100,
    num_trials = 3000,
    # Training
    num_full_train_iters = 100_000,
    keyind = 13,
    orth_u = True,
    fix_output=True,
    retrain_lr = 1e-2,
    retrain_iters = 50_000,
    num_retrain_trials = 1000,
    batch=True,
    batch_size=100,
    input_noise=False,
    input_noise_sigma=0.1
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
    config['fix_output'],
    noise=config['input_noise'],
    noise_sigma=config['input_noise_sigma'])

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
params, losses = fit_mwg_context_nm_rnn(task_samples_in.transpose((0,2,1)), 
                                        context_samples_in.transpose((0,2,1)), 
                                        samples_out.transpose((0,2,1)), 
                                        masks.transpose((0,2,1)),
                                        init_params, 
                                        optimizer, 
                                        x0, 
                                        z0, 
                                        config['num_full_train_iters'],
                                        config['tau_x'], 
                                        config['tau_z'], 
                                        plots=False, 
                                        wandb_log=True, 
                                        final_wandb_plot=True, 
                                        orth_u=config['orth_u'],
                                        batch=config['batch'], 
                                        batch_size=config['batch_size'], 
                                        keyind=config['keyind'])

# log model
log_wandb_model(params, "multitask_context_nmrnn_r{}_n{}_m{}".format(config['R'],config['N'],config['M']), 'model')

# log % correct
# TODO: currently only works for memory anti held-out
samples_in_test = jnp.load('test_inputs.npy')[:75,:6]
samples_out_test = jnp.load('test_outputs.npy')[:75]
samples_labels_test = jnp.load('test_labels.npy')[:75]
x0 = jnp.ones((config['N'],))*0.1
z0 = jnp.ones((config['M'],))*0.1

task_samples_in_test = samples_in_test[:,:-3, :]
context_samples_in_test = samples_in_test[:,-3:, :]

ys_test, _, zs_test = batched_context_nm_rnn(params, x0, z0, task_samples_in_test.transpose((0,2,1)), context_samples_in_test.transpose((0,2,1)), 10, 100, True)

wandb.log({'percent_correct_threetask_test':percent_correct(samples_in_test, samples_out_test, ys_test)})

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

# retraining procedure
task_list = [sample_memory_anti]
task_order, samples_in, samples_out = random_trials(
    jr.PRNGKey(config['keyind']), 
    task_list, 
    config['T'], 
    config['num_retrain_trials'],
    noise=config['input_noise'],
    noise_sigma=config['input_noise_sigma'])

# make new context input
samples_in_heldout = jnp.zeros((config['num_retrain_trials'],7,100))
samples_in_heldout = samples_in_heldout.at[:,:3,:].set(samples_in[:,:3,:]) # sensory inputs are the same
samples_in_heldout = samples_in_heldout.at[:,3:-1,:].set(config['input_noise_sigma']*jr.normal(jr.PRNGKey(config['keyind']),samples_in_heldout[:,3:-1,:].shape)) # add noise to new channels
samples_in_heldout = samples_in_heldout.at[:,-1,:].set(samples_in[:,-1,:]) # add new one-hot input for new task
task_samples_in_heldout = samples_in_heldout[:,:-4, :]
context_samples_in_heldout = samples_in_heldout[:,-4:, :]

# add new dimension to nm_input_weight
params['nm_input_weight'] = jnp.hstack((params['nm_input_weight'], 0.1*jnp.ones((config['M'],1))))

key = jr.PRNGKey(config['keyind'])

# define a simple optimizer
# optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=config['retrain_lr']),
)

x0 = jnp.ones((config['N'],))*0.1
z0 = jnp.ones((config['M'],))*0.1
masks = jnp.ones_like(samples_out)

nm_params = {k: params[k] for k in ('nm_rec_weight', 'nm_input_weight', 'nm_sigmoid_weight', 'nm_sigmoid_intercept')}
lr_params = {k: params[k] for k in ('row_factors', 'column_factors', 'input_weights', 'readout_weights', 'readout_bias')}

params, nm_only_losses = fit_context_nm_only(task_samples_in_heldout.transpose((0,2,1)),
                                             context_samples_in_heldout.transpose((0,2,1)), 
                                             samples_out.transpose((0,2,1)), 
                                             masks.transpose((0,2,1)), 
                                             nm_params,
                                             lr_params, 
                                             optimizer, 
                                             x0, 
                                             z0, 
                                             config['retrain_iters'],
                                             config['tau_x'], 
                                             config['tau_z'], 
                                             wandb_log=True, 
                                             orth_u=config['orth_u'],
                                             batch=config['batch'], 
                                             batch_size=config['batch_size'], 
                                             keyind=config['keyind'])

# log retrained model
log_wandb_model(params, "multitask_context_nmrnn_r{}_n{}_m{}_retrained".format(config['R'],config['N'],config['M']), 'model')

# calc percent correct on trained datapoints
ys_test, _, zs_test = batched_context_nm_rnn(params, x0, z0, task_samples_in_heldout.transpose((0,2,1)), context_samples_in_heldout.transpose((0,2,1)), config['tau_x'], config['tau_z'], True)

wandb.log({'percent_correct_heldouttask_train':percent_correct(samples_in, samples_out, ys_test)})

# calc percent correct on held-out datapoints
samples_in_test = jnp.load('test_inputs.npy')[75:]
samples_out_test = jnp.load('test_outputs.npy')[75:]
samples_labels_test = jnp.load('test_labels.npy')[75:]
x0 = jnp.ones((config['N'],))*0.1
z0 = jnp.ones((config['M'],))*0.1

task_samples_in_test = samples_in_test[:,:-4, :]
context_samples_in_test = samples_in_test[:,-4:, :]

ys_test, _, zs_test = batched_context_nm_rnn(params, x0, z0, task_samples_in_test.transpose((0,2,1)), context_samples_in_test.transpose((0,2,1)), config['tau_x'], config['tau_z'], True)

wandb.log({'percent_correct_heldouttask_test':percent_correct(samples_in_test, samples_out_test, ys_test)})