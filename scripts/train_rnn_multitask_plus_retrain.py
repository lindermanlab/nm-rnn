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
from nmrnn.util import random_rnn_params, log_wandb_model, percent_correct
from nmrnn.fitting import fit_rnn_mwg, fit_rnn_inputweights_only
from nmrnn.rnn_code import batched_rnn, rnn

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
projectname = "multitask-rnn"
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

x0 = jnp.ones((config['N'],))*0.1
masks = jnp.ones_like(samples_out)

# generate random initial parameters
init_params = random_rnn_params(key, config['U'], config['N'], config['O'])

# train on all params
params, losses = fit_rnn_mwg(samples_in.transpose((0,2,1)), 
                                samples_out.transpose((0,2,1)), 
                                masks.transpose((0,2,1)),
                                init_params, 
                                optimizer, 
                                x0, 
                                config['num_full_train_iters'],
                                config['tau'], 
                                wandb_log=True, 
                                batch=config['batch'],
                                batch_size=config['batch_size'])

# log model
log_wandb_model(params, "multitask_context_rnn_n{}".format(config['N']), 'model')

# log % correct
# TODO: currently only works for memory anti held-out
samples_in_test = jnp.load('test_inputs.npy')[:75,:6]
samples_out_test = jnp.load('test_outputs.npy')[:75]
samples_labels_test = jnp.load('test_labels.npy')[:75]
x0 = jnp.ones((config['N'],))*0.1

task_samples_in_test = samples_in_test[:,:-3, :]
context_samples_in_test = samples_in_test[:,-3:, :]

ys_test, _ = batched_rnn(params, x0, samples_in_test.transpose((0,2,1)), config['tau'])

wandb.log({'percent_correct_threetask_test':percent_correct(samples_in_test, samples_out_test, ys_test)})


# example outputs plot
sample_inputs, sample_targets, sample_masks = samples_in.transpose((0,2,1))[0], samples_out.transpose((0,2,1))[0], masks.transpose((0,2,1))[0] # grab a single trial to plot output

ys, _ = rnn(params, x0, sample_inputs, config['tau'])

fig, ax = plt.subplots(1, 1, figsize=[10, 6])
# ax0.xlabel('Timestep')
ax.plot(sample_targets, label='True target')
ax.plot(ys, label='RNN target')
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

params['input_weights'] = jnp.hstack((params['input_weights'], 0.1*jnp.ones((config['N'],1))))

key = jr.PRNGKey(config['keyind'])

# define a simple optimizer
# optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=config['retrain_lr']),
)

x0 = jnp.ones((config['N'],))*0.1
masks = jnp.ones_like(samples_out)

input_params = {k: params[k] for k in ['input_weights']}
other_params = {k: params[k] for k in ('recurrent_weights', 'readout_bias', 'readout_weights')}

params, input_only_losses = fit_rnn_inputweights_only(samples_in_heldout.transpose((0,2,1)),
                                             samples_out.transpose((0,2,1)), 
                                             masks.transpose((0,2,1)), 
                                             input_params,
                                             other_params,
                                             optimizer, 
                                             x0, 
                                             config['retrain_iters'],
                                             config['tau'], 
                                             wandb_log=True,)

log_wandb_model(params, "multitask_context_rnn_n{}_retrained".format(config['N']), 'model')

# calc percent correct on trained datapoints
ys_test, _ = batched_rnn(params, x0, samples_in_heldout.transpose((0,2,1)), config['tau'])

wandb.log({'percent_correct_heldouttask_train':percent_correct(samples_in, samples_out, ys_test)})

# calc percent correct on held-out datapoints
samples_in_test = jnp.load('test_inputs.npy')[75:]
samples_out_test = jnp.load('test_outputs.npy')[75:]
samples_labels_test = jnp.load('test_labels.npy')[75:]
x0 = jnp.ones((config['N'],))*0.1

ys_test, _ = batched_rnn(params, x0, samples_in_test.transpose((0,2,1)), config['tau'])

wandb.log({'percent_correct_heldouttask_test':percent_correct(samples_in_test, samples_out_test, ys_test)})