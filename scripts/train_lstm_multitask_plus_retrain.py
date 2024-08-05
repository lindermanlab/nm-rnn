# this script trains a low-rank RNN in the multitask setting
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

# import matplotlib
# matplotlib.use('TkAgg') # need this on my machine for some reason
import matplotlib.pyplot as plt
import wandb

from nmrnn.data_generation import sample_memory_pro, sample_memory_anti, sample_delay_pro, sample_delay_anti, sample_dm1, sample_dm2, random_trials, one_of_each
from nmrnn.util import random_lstm_params, log_wandb_model, percent_correct
from nmrnn.fitting import fit_lstm_mwg, fit_lstm_inputweights_only
from nmrnn.rnn_code import batched_simple_lstm, simple_lstm, initialize_carry

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    N = 100,    # hidden state dim
    U = 6,      # input dim (3+num_tasks) (BE SURE THIS IS RIGHT)
    O = 3,      # output dimension
    # Model Hyperparameters
    tau = 10,
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
    num_trials = 500,
    # Training
    num_full_train_iters = 100_000,
    keyind = 13,
    fix_output=True,
    retrain_lr = 1e-2,
    retrain_iters = 50_000,
    num_retrain_trials = 1000,
    batch=True,
    batch_size=100,
    input_noise=True,
    input_noise_sigma=0.1
)

# wandb stuff
projectname = "multitask-lstm"
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

key = jr.PRNGKey(config['keyind'])

# define a simple optimizer
# optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=1e-3),
)

# generate random initial parameters
init_params = random_lstm_params(key, config['U'], config['N'], config['O'])
init_carry = initialize_carry(config['N'], key)
# 0.1 initial parameters
c0, h0 = 0.1*jnp.ones_like(init_carry[0]), 0.1*jnp.ones_like(init_carry[1])
init_carry = (c0, h0)

# train on all params
params, losses = fit_lstm_mwg(samples_in.transpose((0,2,1)), 
                                samples_out.transpose((0,2,1)),
                                init_params, 
                                optimizer, 
                                init_carry, 
                                config['num_full_train_iters'], 
                                wandb_log=True, 
                                batch=config['batch'],
                                batch_size=config['batch_size'],
                                keyind=config['keyind'])

# log model
log_wandb_model(params, "multitask_lstm_n{}".format(config['N']), 'model')

# log % correct
# TODO: currently only works for memory anti held-out
samples_in_test = jnp.load('test_inputs.npy')[:75,:6]
samples_out_test = jnp.load('test_outputs.npy')[:75]
samples_labels_test = jnp.load('test_labels.npy')[:75]
x0 = jnp.ones((config['N'],))*0.1

task_samples_in_test = samples_in_test[:,:-3, :]
context_samples_in_test = samples_in_test[:,-3:, :]

_, ys_test = batched_simple_lstm(params, c0, h0, samples_in_test.transpose((0,2,1)))

wandb.log({'percent_correct_threetask_test':percent_correct(samples_in_test, samples_out_test, ys_test)})


# example outputs plot
sample_inputs, sample_targets = samples_in.transpose((0,2,1))[0], samples_out.transpose((0,2,1))[0] # grab a single trial to plot output

ys, _ = simple_lstm(params, c0, h0, sample_inputs)

fig, ax = plt.subplots(1, 1, figsize=[10, 6])
# ax0.xlabel('Timestep')
ax.plot(sample_targets, label='True target')
ax.plot(ys, label='LSTM target')
ax.legend()
ax.set_xlabel('Timestep')
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
samples_in_heldout = jnp.zeros((config['num_retrain_trials'],config['U']+1,config['T']))
samples_in_heldout = samples_in_heldout.at[:,:3,:].set(samples_in[:,:3,:]) # sensory inputs are the same
samples_in_heldout = samples_in_heldout.at[:,3:-1,:].set(config['input_noise_sigma']*jr.normal(jr.PRNGKey(config['keyind']),samples_in_heldout[:,3:-1,:].shape)) # add noise to new channels
samples_in_heldout = samples_in_heldout.at[:,-1,:].set(samples_in[:,-1,:]) # add new one-hot input for new task

params['weights_iu'] = jnp.hstack((params['weights_iu'], 0.1*jnp.ones((config['N'],1))))
params['weights_fu'] = jnp.hstack((params['weights_fu'], 0.1*jnp.ones((config['N'],1))))
params['weights_gu'] = jnp.hstack((params['weights_gu'], 0.1*jnp.ones((config['N'],1))))
params['weights_ou'] = jnp.hstack((params['weights_ou'], 0.1*jnp.ones((config['N'],1))))

key = jr.PRNGKey(config['keyind'])

# define a simple optimizer
# optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=config['retrain_lr']),
)

input_params = {k: params[k] for k in ('weights_iu', 'weights_fu', 'weights_gu', 'weights_ou')}
other_params = {k: params[k] for k in ('weights_ih', 'bias_ih', 'weights_fh', 'bias_fh', 'weights_gh', 'bias_gh', 'weights_oh', 'bias_oh', 'readout_weights')}

params, input_only_losses = fit_lstm_inputweights_only(samples_in_heldout.transpose((0,2,1)),
                                             samples_out.transpose((0,2,1)), 
                                             input_params,
                                             other_params,
                                             optimizer, 
                                             init_carry, 
                                             config['retrain_iters'],
                                             wandb_log=True,
                                             batch=config['batch'], 
                                             batch_size=config['batch_size'], 
                                             keyind=config['keyind'])

log_wandb_model(params, "multitask_context_lstm_n{}_retrained".format(config['N']), 'model')

# calc percent correct on trained datapoints
_, ys_test = batched_simple_lstm(params, c0, h0, samples_in_heldout.transpose((0,2,1)))

wandb.log({'percent_correct_heldouttask_train':percent_correct(samples_in, samples_out, ys_test)})

# calc percent correct on held-out datapoints
samples_in_test = jnp.load('test_inputs.npy')[75:]
samples_out_test = jnp.load('test_outputs.npy')[75:]
samples_labels_test = jnp.load('test_labels.npy')[75:]

_, ys_test = batched_simple_lstm(params, c0, h0, samples_in_test.transpose((0,2,1)))

wandb.log({'percent_correct_heldouttask_test':percent_correct(samples_in_test, samples_out_test, ys_test)})