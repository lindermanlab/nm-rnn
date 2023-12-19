import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import wandb

from nmrnn.data_generation import sample_all
from nmrnn.util import random_nmrnn_params, log_wandb_model
from nmrnn.fitting import fit_mwg_nm_rnn, fit_mwg_nm_only
from nmrnn.rnn_code import batched_nm_rnn

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
wandb.init(config=default_config, project=projectname, entity='nm-rnn')
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

# generate random initial parameters
init_params = random_nmrnn_params(key, config['U'], config['N'], config['R'],
                                  config['M'], config['K'], config['O'])

# split parameters for now (only train on nm params to start)
nm_params = {k: init_params[k] for k in ('readout_weights', 'nm_rec_weight', 'nm_input_weight', 'nm_sigmoid_weight', 'nm_sigmoid_intercept')}
other_params = {k: init_params[k] for k in ('row_factors', 'column_factors', 'input_weights')}

# train on nm params only for a bit
params, nm_only_losses = fit_mwg_nm_only(all_inputs, all_outputs, all_masks, nm_params,
                                 other_params, optimizer, x0, z0, config['num_nm_only_iters'],
                                         config['tau_x'], config['tau_z'], plots=False, wandb_log=True, final_wandb_plot=False)

# train on all params
params, losses = fit_mwg_nm_rnn(all_inputs, all_outputs, all_masks,
                                params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'], plots=False, wandb_log=True, final_wandb_plot=True)

# log model
log_wandb_model(params, "nmrnn_r{}_n{}_m{}".format(config['R'],config['N'],config['M']), 'model')

# more plots/metrics to analyze model generalization

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

ys, _, zs = batched_nm_rnn(params, x0, z0, new_inputs, config['tau_x'], config['tau_z'])

# plot aligned to start
m = params['nm_sigmoid_weight']
b = params['nm_sigmoid_intercept']

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=[10,6])

ax0.set_title('NM signals aligned to time 0')
for i in range(12):
    ax0.plot(jax.nn.sigmoid((zs @ m.T + b)[i,:,0]))

for i in range(12):
    ax1.plot(jax.nn.sigmoid((zs @ m.T + b)[i,:,1]))

for i in range(12):
    ax2.plot(jax.nn.sigmoid((zs @ m.T + b)[i,:,2]))

wandb.log({'nm_aligned_0':wandb.Image(fig)}, commit=True)

# plot aligned to go cue
go_cues = jnp.where(new_inputs[:,:,2])[1]
go_mask = jnp.zeros((12, 110), dtype=bool)
ind_range = jnp.arange(110)

for i in range(12):
    go_mask = go_mask.at[i].set((ind_range > go_cues[i] - 30) * (ind_range < go_cues[i] + 60))

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=[10,6])

ax0.set_title('NM signals aligned to Go cue')
for i in range(12):
    ax0.plot(jax.nn.sigmoid((zs @ m.T + b)[i,:,0])[go_mask[i]])

for i in range(12):
    ax1.plot(jax.nn.sigmoid((zs @ m.T + b)[i,:,1])[go_mask[i]])

for i in range(12):
    ax2.plot(jax.nn.sigmoid((zs @ m.T + b)[i,:,2])[go_mask[i]])

wandb.log({'nm_aligned_go':wandb.Image(fig)}, commit=True)

# single split plot also showing desired output
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=[3, 5], sharex=True)

measure = jnp.where(new_inputs[6,:,0]>0)
wait = jnp.where(new_inputs[6,:,1]>0)
go = jnp.where(new_inputs[6,:,2]>0)
ramp_end = jnp.min(jnp.where(new_outputs[6,:,:]==0.5)[0])

ax0.plot(new_outputs[6,:,0], c='k')
ax0.axvline(x=measure, c='k', ls='--', alpha=0.7)
ax0.axvline(x=wait, c='k', ls='--', alpha=0.7)
ax0.axvline(x=go, c='k', ls='--', alpha=0.7)
ax0.axvline(x=ramp_end, c='k', ls='--', alpha=0.7)

ax1.plot(jax.nn.sigmoid((zs @ m.T + b)[6,:,0].T), c='b', lw=2)
ax1.axvline(x=measure, c='k', ls='--', alpha=0.7)
ax1.axvline(x=wait, c='k', ls='--', alpha=0.7)
ax1.axvline(x=go, c='k', ls='--', alpha=0.7)
ax1.axvline(x=ramp_end, c='k', ls='--', alpha=0.7)

ax2.plot(jax.nn.sigmoid((zs @ m.T + b)[6,:,1].T), c='b', lw=2)
ax2.axvline(x=measure, c='k', ls='--', alpha=0.7)
ax2.axvline(x=wait, c='k', ls='--', alpha=0.7)
ax2.axvline(x=go, c='k', ls='--', alpha=0.7)
ax2.axvline(x=ramp_end, c='k', ls='--', alpha=0.7)

ax3.plot(jax.nn.sigmoid((zs @ m.T + b)[6,:,2].T), c='b', lw=2)
ax3.axvline(x=measure, c='k', ls='--', alpha=0.7)
ax3.axvline(x=wait, c='k', ls='--', alpha=0.7)
ax3.axvline(x=go, c='k', ls='--', alpha=0.7)
ax3.axvline(x=ramp_end, c='k', ls='--', alpha=0.7)

wandb.log({'single_output':wandb.Image(fig)}, commit=True)
