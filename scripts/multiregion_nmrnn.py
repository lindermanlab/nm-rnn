import jax
import jax.numpy as jnp
import jax.random as jr
import math

from jax import grad, vmap, jit
from jax import lax
import optax
import matplotlib.pyplot as plt
import wandb

from nmrnn.util import log_wandb_model

def init_params(key, n_bg, n_nm, g_bg, g_nm, input_dim, output_dim):
    # for now assume Th/BG/C are same size, g is the same for all weight matrices
    skeys = jr.split(key, 17)

    # bg parameters
    J_bg = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[0], (n_bg, n_bg))
    B_bgc = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[1], (n_bg, n_bg))

    # c parameters
    J_c = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[2], (n_bg, n_bg))
    B_cu = (1 / math.sqrt(input_dim)) * jr.normal(skeys[3], (n_bg, input_dim))
    B_ct = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[4], (n_bg, n_bg))

    # t parameters
    J_t = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[5], (n_bg, n_bg))
    B_tbg = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[6], (n_bg, n_bg))

    # nm parameters
    J_nm = (g_nm / math.sqrt(n_nm)) * jr.normal(skeys[7], (n_nm, n_nm))
    J_nmc = (g_nm / math.sqrt(n_nm)) * jr.normal(skeys[8], (n_nm, n_bg))
    B_nmc = (1 / math.sqrt(n_nm)) * jr.normal(skeys[9], (n_nm, n_bg))

    m = (1 / math.sqrt(n_nm)) * jr.normal(skeys[10], (1, n_nm))
    c = (1 / math.sqrt(n_nm)) * jr.normal(skeys[11])

    U = (1 / math.sqrt(n_bg)) * jr.normal(skeys[12], (1, n_bg))
    V_bg = (1 / math.sqrt(n_bg)) * jr.normal(skeys[13], (1, n_bg))
    V_c = (1 / math.sqrt(n_bg)) * jr.normal(skeys[14], (1, n_bg))

    # readout params
    C = (1 / math.sqrt(n_bg)) * jr.normal(skeys[15], (output_dim, n_bg))
    rb = (1 / math.sqrt(n_bg)) * jr.normal(skeys[16], (output_dim, ))

    return {
        'J_bg': J_bg,
        'B_bgc': B_bgc,
        'J_c': J_c,
        'B_cu': B_cu,
        'B_ct': B_ct,
        'J_t': J_t,
        'B_tbg': B_tbg,
        'J_nm': J_nm,
        'J_nmc': J_nmc,
        'B_nmc': B_nmc,
        'm': m,
        'c': c,
        'C': C,
        'rb': rb,
        'U': U,
        'V_bg': V_bg,
        'V_c': V_c
    }

def multiregion_nmrnn(params, x_0, z_0, inputs, tau_x, tau_z, modulation=True):
    """
    Arguments:
    - params
    - x0
    - inputs
    - tau   : decay constant
    """

    # unpack params (TODO)
    x_bg0, x_c0, x_t0 = x_0
    x_nm0 = z_0

    J_bg = params['J_bg']
    # J_bg, _ = jnp.linalg.qr(J_bg)
    B_bgc = params['B_bgc']
    J_c = params['J_c']
    # J_c, _ = jnp.linalg.qr(J_c)
    B_cu = params['B_cu']
    B_ct = params['B_ct']
    J_t = params['J_t']
    # J_t, _ = jnp.linalg.qr(J_t)
    B_tbg = params['B_tbg']
    J_nm = params['J_nm']
    B_nmc = params['B_nmc']
    m = params['m']
    c = params['c']
    C = params['C']
    rb = params['rb']

    tau_c = tau_x
    tau_bg = tau_x
    tau_t = tau_x
    tau_nm = tau_z

    def nln(x):
        # return jnp.maximum(0, x)
        return jnp.tanh(x)

    def exc(w):
        return jnp.abs(w)

    def inh(w):
        return -jnp.abs(w)

    def _step(x_and_z, u):
        x_bg, x_c, x_t, x_nm = x_and_z
        # could also parameterize weight matrices via singular values (passed through sigmoid)
        # thalamus: diagonal recurrence?

        # update x_c
        x_c_new = (1.0 - (1. / tau_c)) * x_c + (1. / tau_c) * J_c @ nln(x_c) # recurrent dynamics
        x_c_new += (1. / tau_c) * B_cu @ u # external inputs
        x_c_new += (1. / tau_c) * exc(B_ct) @ x_t # input from thalamus, excitatory

        # update x_bg
        num_bg_cells = J_bg.shape[0]
        num_c_cells = J_c.shape[0]

        if modulation:
            #TODO: will error out if num_bg_cells isn't even / add initializations
            U = jnp.concatenate((jnp.ones((num_bg_cells//2, 1)), -1*jnp.ones((num_bg_cells//2, 1)))) # direct/indirect
            V_bg = jnp.ones((num_bg_cells, 1))
            V_c = jnp.ones((num_c_cells, 1))
            s = jax.nn.sigmoid(m @ x_nm + c) # neuromodulatory signal (1D for now)
            G_bg = jnp.exp(s * U @ V_bg.T) # TODO: change to matrix U,V + vector s (for multidimensional NM)
            G_c = jnp.exp(s * U @ V_c.T) # num_bg_cells x num_c_cells
        else:
            G_bg = jnp.ones((num_bg_cells, num_bg_cells))
            G_c = jnp.ones((num_bg_cells, num_c_cells))

        x_bg_new = (1.0 - (1. / tau_bg)) * x_bg + (1. / tau_bg) * (G_bg * inh(J_bg)) @ nln(x_bg) # recurrent dynamics, inhibitory
        x_bg_new += (1. / tau_bg) * (G_c * exc(B_bgc)) @ x_c # input from cortex, excitatory

        # update x_t
        x_t_new = (1.0 - (1. / tau_t)) * x_t # remove recurrent dynamics
        # x_t_new = (1.0 - (1. / tau_t)) * x_t + (1. / tau_t) * J_t @ nln(x_t) # recurrent dynamics
        B_tbg_eff = jnp.concatenate((exc(B_tbg[:, :num_bg_cells//2]), inh(B_tbg[:, num_bg_cells//2:])), axis=1)
        x_t_new += (1. / tau_t) * B_tbg_eff @ x_bg # input from BG, excitatory/inhibitory
        # x_t += (1. / tau_t) * inh(B_tbg) @ x_bg # input from BG, inhibitory

        # update x_nm
        if modulation:
            x_nm_new = (1.0 - (1. / tau_nm)) * x_nm + (1. / tau_nm) * J_nm @ nln(x_nm)
            x_nm_new += (1. / tau_nm) * exc(B_nmc) @ x_c # input from cortex, excitatory
        else:
            x_nm_new = x_nm

        # calculate y
        C_eff = jnp.concatenate((exc(C[:, :num_bg_cells//2]), inh(C[:, num_bg_cells//2:])), axis=1)
        y = C_eff @ x_bg + rb # output from BG
        return (x_bg_new, x_c_new, x_t_new, x_nm_new), (y, x_bg_new, x_c_new, x_t_new, x_nm_new)

    _, (y, x_bg, x_c, x_t, x_nm) = lax.scan(_step, (x_bg0, x_c0, x_t0, x_nm0), inputs)

    return y, (x_bg, x_c, x_t), x_nm

batched_nm_rnn = vmap(multiregion_nmrnn, in_axes=(None, None, None, 0, None, None, None))

def batched_nm_rnn_loss(params, x0, z0, batch_inputs, tau_x, tau_z, batch_targets, batch_mask, modulation=True):
    ys, _, _ = batched_nm_rnn(params, x0, z0, batch_inputs, tau_x, tau_z, modulation) # removed orth_u from here
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)

def fit_mwg_nm_rnn(inputs, targets, loss_masks, params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                   wandb_log=False, modulation=True, log_interval=200): # training on full set of data
    opt_state = optimizer.init(params)
    N_data = inputs.shape[0]

    @jit
    def _step(params_and_opt, input):
        (params, opt_state) = params_and_opt
        loss_value, grads = jax.value_and_grad(batched_nm_rnn_loss)(params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks, modulation=modulation)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []

    best_loss = 1e6
    best_params = params
    for n in range(num_iters//log_interval):
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=log_interval) #arange bc the inputs aren't changing
        losses.append(loss_values)
        print(f'step {(n+1)*200}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss:
            best_params = params
            best_loss = loss_values[-1]

    return best_params, losses

def mwg_tasks(T,
               intervals,
               measure_min,
               measure_max,
               delay,
               mask_pad=None,
               output_min=-0.5,
               output_max=0.5):
    """
    Simulates all possible input/output pairs for the measure-set-go task.

    Arguments:
    T: length of the trial
    intervals: (num_contexts, num_intervals) array of intervals for each contet
    measure_min: minimum time when measure cue comes
    measure_max: maximum time when measure cue comes
    delay: length of time between measure and go

    Returns:
    inputs: (T, 3) array of time series (measure, set, go, tonic cue) where
            measure, set, go are one-hot time series marking each event

    output: (T, 1) ramp starting at go and lasting (set - measure) time

    mask: (T, 1) binary mask over ramp +- 300 units of time

    Note: outputs are aligned at *go* cue
    """
    num_contexts, num_intervals = intervals.shape

    def _single(context_ind, interval_ind, t_measure):
        # construct the input array
        choice_intervals = intervals[context_ind][0]
        max_interval = jnp.max(intervals)
        this_interval = choice_intervals[interval_ind]
        t_set = t_measure + this_interval
        t_go = t_set + delay
        inputs = jnp.column_stack([
            10*(jnp.arange(T) == t_measure),
            10*(jnp.arange(T) == t_set),
            10*(jnp.arange(T) == t_go),
            # context_ind * jnp.ones(T)
        ])

        # construct the outputs
        slope = 1.0 / this_interval
        intercept = -t_go * slope
        outputs = jnp.clip(intercept + jnp.arange(T) * slope, 0, 1)
        outputs = output_min + (output_max - output_min) * outputs

        # construct the mask
        if mask_pad is not None: mask = (jnp.arange(T) >= t_go-mask_pad) & (jnp.arange(T) < t_go + max_interval + mask_pad)
        else: mask = jnp.ones(T)

        return inputs, outputs, mask

    context_inds = jnp.arange(num_contexts)
    interval_inds = jnp.arange(num_intervals)
    possible_t_measure = jnp.arange(measure_min, measure_max)

    batch_contexts = jnp.repeat(context_inds, num_intervals*(measure_max-measure_min))[:,None]
    batch_intervals = jnp.repeat(interval_inds, num_contexts*(measure_max-measure_min))[:,None]
    batch_t_measures = jnp.tile(possible_t_measure, num_contexts*num_intervals)[:,None]

    all_inputs, all_outputs, all_masks, = jax.vmap(_single)(batch_contexts, batch_intervals, batch_t_measures)

    return all_inputs, all_outputs[:,:,None], all_masks[:,:,None]

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    n_bg = 20,
    n_nm = 5,      # NM (SNc) dimension
    g_bg = 1.4,
    g_nm = 1.4,
    U = 3,      # input dim
    O = 1,      # output dimension
    # Model Hyperparameters
    tau_x = 10,
    tau_z = 100,
    modulation = True,
    # Timing (task) parameters
    dt = 10, # ms
    # Data Generation
    T = 110,
    measure_min = 10,
    measure_max = 20,
    intervals = [[12, 14, 16, 18]],
    delay = 15,
    # Training
    num_nm_only_iters = 0,
    num_full_train_iters = 4000,
    keyind = 13,
)

projectname = "multiregion-nm-rnn"
wandb.init(config=default_config, project=projectname, entity='nm-rnn')
config = wandb.config

all_inputs, all_outputs, all_masks = mwg_tasks(config['T'],
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

x_bg0 = jnp.ones((config['n_bg'],)) * 0.1
x_c0 = jnp.ones((config['n_bg'],)) * 0.1
x_t0 = jnp.ones((config['n_bg'],)) * 0.1
x0 = (x_bg0, x_c0, x_t0)
z0 = jnp.ones((config['n_nm'],)) * 0.1

# generate random initial parameters
params = init_params(
    key,
    config['n_bg'], config['n_nm'],
    config['g_bg'], config['g_nm'],
    config['U'], config['O']
)

# split parameters for now (only train on nm params to start)
# nm_params = {k: params[k] for k in ('readout_weights', 'nm_rec_weight', 'nm_input_weight', 'nm_sigmoid_weight', 'nm_sigmoid_intercept')}
# other_params = {k: params[k] for k in ('row_factors', 'column_factors', 'input_weights')}

# train on all params
params_nm, losses_nm = fit_mwg_nm_rnn(all_inputs, all_outputs, all_masks,
                                params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'], wandb_log=True, modulation=config['modulation'])

log_wandb_model(params_nm, "multiregion_nmrnn_n{}_m{}_mod{}".format(config['n_bg'], config['n_nm'], config['modulation']), 'model')

all_inputs, all_outputs, all_masks = mwg_tasks(config['T'],
           jnp.array([[8, 10, 12, 14, 16, 18, 20, 22]]),
            15, 16,
            config['delay'],)

def align_to_go(data, inputs):
    """
    align data to go cue
    data: shape (8, 110, N) or (8, 110)
    return: shape (8, 60, N) or (8, 60)
    """
    go_cues = jnp.where(inputs[:,:,2])[1]
    go_mask = jnp.zeros((12, 110), dtype=bool)
    ind_range = jnp.arange(110)

    new_data = []
    for i in range(8):
        go_mask = (ind_range > go_cues[i] - 20) * (ind_range < go_cues[i] + 40)
        new_data.append(data[i, go_mask])
    new_data = jnp.stack(new_data)

    return new_data

fig, ax = plt.subplots(4, 1, figsize=(3,8))
ys, xs, zs = batched_nm_rnn(params_nm, x0, z0, all_inputs, config['tau_x'], config['tau_z'], config['modulation'])
for idx, name in enumerate(['BG', 'Cortex', 'Thalamus', 'SNc']):
    mean_act = jnp.mean(xs[idx], axis=2) if idx < 3 else jnp.mean(zs, axis=2)
    ax[idx].axvline(x=15, c='k', ls='--', alpha=0.7)
    ax[idx].plot(mean_act[:2].T, c='tab:red')
    ax[idx].plot(mean_act[2:6].T, c='tab:purple')
    ax[idx].plot(mean_act[6:].T, c='tab:blue')
    ax[idx].set_title(name + ' mean activity')

plt.suptitle('aligned to measure')
plt.tight_layout()
wandb.log({'four_regions_0':wandb.Image(fig)}, commit=True)

fig, ax = plt.subplots(4, 1, figsize=(3,8))
for idx, name in enumerate(['BG', 'Cortex', 'Thalamus', 'SNc']):
    mean_act = jnp.mean(xs[idx], axis=2) if idx < 3 else jnp.mean(zs, axis=2)
    mean_act = align_to_go(mean_act, all_inputs)

    ax[idx].plot(mean_act[:2].T, c='tab:red')
    ax[idx].plot(mean_act[2:6].T, c='tab:purple')
    ax[idx].plot(mean_act[6:].T, c='tab:blue')
    ax[idx].axvline(x=4, c='k', ls='--', alpha=0.7)
    ax[idx].axvline(x=19, c='k', ls='--', alpha=0.7)
    ax[idx].set_title(name + ' mean activity')

plt.suptitle('aligned to go')
plt.tight_layout()
wandb.log({'four_regions_go':wandb.Image(fig)}, commit=True)

fig, ax = plt.subplots(1, 2, figsize=(6,3))
for idx, name in enumerate(['BG (direct)', 'BG (indirect)']):
    mean_act = jnp.mean(xs[0][:,:,10*idx:10*(idx+1)], axis=2)
    ax[idx].axvline(x=15, c='k', ls='--', alpha=0.7)
    ax[idx].plot(mean_act[:2].T, c='tab:red')
    ax[idx].plot(mean_act[2:6].T, c='tab:purple')
    ax[idx].plot(mean_act[6:].T, c='tab:blue')
    ax[idx].set_title(name + ' mean activity')

plt.suptitle('aligned to measure')
plt.tight_layout()
wandb.log({'bg_split_0':wandb.Image(fig)}, commit=True)

fig, ax = plt.subplots(1, 2, figsize=(6,3))
for idx, name in enumerate(['BG (direct)', 'BG (indirect)']):
    mean_act = jnp.mean(xs[0][:,:,10*idx:10*(idx+1)], axis=2)
    mean_act = align_to_go(mean_act, all_inputs)

    ax[idx].axvline(x=4, c='k', ls='--', alpha=0.7)
    ax[idx].axvline(x=19, c='k', ls='--', alpha=0.7)
    ax[idx].plot(mean_act[:2].T, c='tab:red')
    ax[idx].plot(mean_act[2:6].T, c='tab:purple')
    ax[idx].plot(mean_act[6:].T, c='tab:blue')
    ax[idx].set_title(name + ' mean activity')

plt.suptitle('aligned to go')
plt.tight_layout()
wandb.log({'bg_split_go':wandb.Image(fig)}, commit=True)


fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(all_outputs[:,:,0].T, linestyle='--', c='k', alpha=0.5)
plt.plot(ys[:2,:,0].T, c='tab:red')
plt.plot(ys[2:6,:,0].T, c='tab:purple')
plt.plot(ys[6:,:,0].T, c='tab:blue')
plt.title('outputs')
wandb.log({'outputs':wandb.Image(fig)}, commit=True)

ys, xs, zs = batched_nm_rnn(params_nm, x0, z0, all_inputs, config['tau_x'], config['tau_z'], modulation=False)
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(all_outputs[:,:,0].T, linestyle='--', c='k', alpha=0.5)
plt.plot(ys[:2,:,0].T, c='tab:red')
plt.plot(ys[2:6,:,0].T, c='tab:purple')
plt.plot(ys[6:,:,0].T, c='tab:blue')
plt.title('outputs (remove modulation)')
wandb.log({'outputs_remove_nm':wandb.Image(fig)}, commit=True)


# plot J_bg and G_bg, G_c