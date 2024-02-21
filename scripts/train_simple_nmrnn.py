import jax
import jax.numpy as jnp
import jax.random as jr
import optax

# import matplotlib
# matplotlib.use('TkAgg') # need this on my machine for some reason
import matplotlib.pyplot as plt
import wandb

from nmrnn.data_generation import sample_all
from nmrnn.util import random_nmrnn_params, log_wandb_model
from nmrnn.fitting import fit_lin_sym_nm_rnn
from nmrnn.rnn_code import batched_lin_sym_nm_rnn

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    N = 100,    # hidden state dim
    R = 3,      # rank of RNN
    U = 3,      # input dim
    O = 1,      # output dimension
    M = 5,      # NM dimension
    # got rid of K for now, set to R by default
    #K = 2,      # NM sigmoid dimension (must be 1 or R)
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
    num_lr_only_iters = 10_000, # if 0, skip lr-only training step
    num_nm_only_iters = 10_000, # if 0, skip nm-only training step
    num_full_train_iters = 50_000,
    keyind = 13,
)

# wandb stuff
projectname = "nm-rnn-mwg"
wandb.init(config=default_config, project=projectname, entity='nm-rnn')
config = wandb.config

# data generation
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
  optax.adamw(learning_rate=1e-3),
)

x0 = jnp.ones((config['N'],))*0.1
z0 = jnp.ones((config['M'],))*0.1

# generate random initial parameters
init_params = random_nmrnn_params(key, config['U'], config['N'], config['R'],
                                  config['M'], config['R'], config['O'])

# train on all params
params, losses = fit_lin_sym_nm_rnn(all_inputs, all_outputs, all_masks,
                                init_params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'])

# log model
log_wandb_model(params, "nmrnn_r{}_n{}_m{}".format(config['R'],config['N'],config['M']), 'model')

################## more plots/metrics to analyze model generalization

# generate data for all intervals (4 trained, plus 4 shorter/longer)
new_intervals = jnp.array([[4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]])
# middling cue time (but could change this)
measure_min = 15
measure_max = 16

new_inputs, new_outputs, new_masks = sample_all(config['T'],
           new_intervals,
            measure_min,
            measure_max,
            config['delay'],)

ys, _, zs = batched_lin_sym_nm_rnn(params, x0, z0, new_inputs, config['tau_x'], config['tau_z'])

################## plot aligned to start
m = params['nm_sigmoid_weight']
b = params['nm_sigmoid_intercept']

fig, axes = plt.subplots(config['R'], 1, figsize=[10,config['R']*2])

for r, ax in enumerate(axes):
    for i in range(12):
        ax.plot(jax.nn.sigmoid((zs @ m.T + b)[i, :, r]))

wandb.log({'nm_aligned_0':wandb.Image(fig)}, commit=True)

################## plot aligned to go cue
go_cues = jnp.where(new_inputs[:,:,2])[1]
go_mask = jnp.zeros((12, 110), dtype=bool)
ind_range = jnp.arange(110)

for i in range(12):
    go_mask = go_mask.at[i].set((ind_range > go_cues[i] - 30) * (ind_range < go_cues[i] + 60))

fig, axes = plt.subplots(config['R'], 1, figsize=[10,config['R']*2])

for r, ax in enumerate(axes):
    for i in range(12):
        ax.plot(jax.nn.sigmoid((zs @ m.T + b)[i, :, r])[go_mask[i]])

wandb.log({'nm_aligned_go':wandb.Image(fig)}, commit=True)

################## single split plot also showing desired output
fig, axes = plt.subplots(config['R'] + 1, 1, figsize=[10,config['R']*2+2], sharex=True)

measure = jnp.where(new_inputs[6,:,0]>0)
wait = jnp.where(new_inputs[6,:,1]>0)
go = jnp.where(new_inputs[6,:,2]>0)
ramp_end = jnp.min(jnp.where(new_outputs[6,:,:]==0.5)[0])

for i, ax in enumerate(axes):
    if i == 0:
        ax.plot(new_outputs[6,:,0], c='k')
        ax.axvline(x=measure, c='k', ls='--', alpha=0.7)
        ax.axvline(x=wait, c='k', ls='--', alpha=0.7)
        ax.axvline(x=go, c='k', ls='--', alpha=0.7)
        ax.axvline(x=ramp_end, c='k', ls='--', alpha=0.7)
    else:
        ax.plot(jax.nn.sigmoid((zs @ m.T + b)[6,:,i-1].T), c='b', lw=2)
        ax.axvline(x=measure, c='k', ls='--', alpha=0.7)
        ax.axvline(x=wait, c='k', ls='--', alpha=0.7)
        ax.axvline(x=go, c='k', ls='--', alpha=0.7)
        ax.axvline(x=ramp_end, c='k', ls='--', alpha=0.7)

wandb.log({'single_output':wandb.Image(fig)}, commit=True)

################## generalization plots
N = new_intervals.shape[1]
# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.brg(jnp.linspace(0,1,N)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharey=True, sharex=True)
for i in [4, 5, 6, 7]:
    ax1.plot(new_outputs[i][new_masks[i].astype(bool)], 'k--', alpha=0.7)
    ax1.plot(ys[i][new_masks[i].astype(bool)], c='tab:purple')

for i in [0, 1, 2, 3]:
    ax2.plot(new_outputs[i][new_masks[i].astype(bool)], 'k--', alpha=0.7)
    ax2.plot(ys[i][new_masks[i].astype(bool)], c='tab:red')

for i in [8,9,10,11]:
    ax2.plot(new_outputs[i][new_masks[i].astype(bool)], 'k--', alpha=0.7)
    ax2.plot(ys[i][new_masks[i].astype(bool)], c='tab:blue')

wandb.log({'generalization_output': wandb.Image(fig)}, commit=True)