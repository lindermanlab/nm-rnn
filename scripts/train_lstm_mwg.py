import jax
import jax.numpy as jnp
import jax.random as jr
import optax

# import matplotlib
# matplotlib.use('TkAgg') # need this on my machine for some reason
import matplotlib.pyplot as plt
import wandb

from nmrnn.data_generation import sample_all
from nmrnn.util import random_lstm_params, log_wandb_model
from nmrnn.fitting import fit_lstm_mwg
from nmrnn.rnn_code import batched_simple_lstm, initialize_carry

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    N = 100,    # hidden size
    U = 3,      # input size
    O = 1,      # output size
    # Timing (task) parameters
    dt = 10,#ms
    # Data Generation
    T = 110,
    measure_min = 10,
    measure_max = 20,
    intervals = [[12, 14, 16, 18]],
    delay = 15,
    # Training
    num_full_train_iters = 10_000,
    keyind = 13,
)

projectname = "lstm-mwg"
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

# generate random initial parameters
init_params = random_lstm_params(key, config['U'], config['N'], config['O'])
init_carry = initialize_carry(config['N'], key)

# train on all params
params, losses = fit_lstm_mwg(all_inputs, all_outputs, init_params, optimizer, 
                              init_carry, config['num_full_train_iters'], wandb_log=True)

# log model
log_wandb_model(params, "lstm_n{}".format(config['N']), 'model')

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

(c0, h0) = init_carry
carrys, ys = batched_simple_lstm(params, c0, h0, new_inputs)

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