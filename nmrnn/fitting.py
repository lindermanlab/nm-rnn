import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax
# import matplotlib
# matplotlib.use('TkAgg') # need these on my machine for some reason
import matplotlib.pyplot as plt
import wandb

from nmrnn.data_generation import sample_one
from nmrnn.rnn_code import batched_nm_rnn_loss, nm_rnn, batched_lr_rnn_loss, lr_rnn, \
    batched_nm_rnn_loss_frozen, lin_sym_nm_rnn, batched_lin_sym_nm_rnn_loss, context_nm_rnn, \
        batched_context_nm_rnn_loss, batched_context_nm_rnn_loss_frozen,batched_lr_rnn_loss_split, \
            lstm_batched_loss, batched_rnn_loss

#TODO: add batching
def fit_mwg_nm_rnn(inputs, targets, loss_masks, params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                   wandb_log=False, orth_u=True): # training on full set of data
    opt_state = optimizer.init(params)
    N_data = inputs.shape[0]

    @jit
    def _step(params_and_opt, input):
        (params, opt_state) = params_and_opt
        loss_value, grads = jax.value_and_grad(batched_nm_rnn_loss)(params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks, orth_u=orth_u)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []

    best_loss = 1e6
    best_params = params
    for n in range(num_iters//1000):
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]

    return best_params, losses


def fit_mwg_lr_rnn(inputs, targets, loss_masks, params, optimizer, x0, num_iters, tau, 
                   wandb_log=False, orth_u=True, batch=False, batch_size=100, keyind=13): # training on full set of data
    opt_state = optimizer.init(params)
    N_data = inputs.shape[0]
    key = jr.PRNGKey(keyind)

    @jit
    def _step(params_and_opt, input):
        (params, opt_state, key) = params_and_opt
        if batch: 
            key, _ = jr.split(key, 2)
            batch_inds = jr.choice(key, jnp.arange(N_data), shape=(batch_size,))
            batch_inputs = inputs[batch_inds]
            batch_targets = targets[batch_inds]
            batch_masks = loss_masks[batch_inds]
        else:
            batch_inputs = inputs
            batch_targets = targets
            batch_masks = loss_masks
        loss_value, grads = jax.value_and_grad(batched_lr_rnn_loss)(params, x0, batch_inputs, tau, batch_targets, batch_masks, orth_u=orth_u)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, key), (params, loss_value)

    losses = []
   
    best_loss = 1e6
    best_params = params
    for n in range(num_iters//1000):
        (params,_,_), (_, loss_values) = lax.scan(_step, (params, opt_state, key), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})

    return best_params, losses


# training function to fit only neuromodulatory parameters
def fit_mwg_nm_only(inputs, targets, loss_masks, nm_params, other_params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                    plots=True, wandb_log=False, final_wandb_plot=False, orth_u=True): # training on full set of data
    opt_state = optimizer.init(nm_params)
    N_data = inputs.shape[0]

    @jit
    def _step(params_and_opt, input):
        (nm_params, opt_state) = params_and_opt
        #pdb.set_breakpoint()
        loss_value, grads = jax.value_and_grad(batched_nm_rnn_loss_frozen)(nm_params, other_params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks, orth_u=orth_u)
        updates, opt_state = optimizer.update(grads, opt_state, nm_params)
        nm_params = optax.apply_updates(nm_params, updates)
        return (nm_params, opt_state), (nm_params, loss_value)

    losses = []
    sample_inputs, sample_targets, sample_masks = inputs[0], targets[0], loss_masks[0]  # grab a single trial to plot output
    
    best_loss = 1e6
    best_nm_params = nm_params

    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (nm_params,_), (_, loss_values) = lax.scan(_step, (nm_params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        # if n % 100 == 0:
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_nm_params = nm_params
            best_loss = loss_values[-1]

        params = dict(nm_params, **other_params)
        ys, _, zs = nm_rnn(params, x0, z0, sample_inputs, tau_x, tau_z, orth_u=orth_u)

        if plots:
            # plt.figure(figsize=[10,6])
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=[10,6])
            # ax0.xlabel('Timestep')
            ax0.plot(sample_targets, label='True target')
            ax0.plot(ys, label='RNN target')
            ax0.legend()
            ax1.set_xlabel('Timestep')
            m = params['nm_sigmoid_weight']
            b = params['nm_sigmoid_intercept']
            ax1.plot(jax.nn.sigmoid((zs @ m.T + b)))
            # ax1.legend()
            plt.pause(0.1)

    if final_wandb_plot:
        # plt.figure(figsize=[10,6])
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=[10, 6])
        # ax0.xlabel('Timestep')
        ax0.set_title('after training nm only')
        ax0.plot(sample_targets, label='True target')
        ax0.plot(ys, label='RNN target')
        ax0.legend()
        ax1.set_xlabel('Timestep')
        m = params['nm_sigmoid_weight']
        b = params['nm_sigmoid_intercept']
        ax1.plot(jax.nn.sigmoid((zs @ m.T + b)))
        wandb.log({'nm_only_curves':wandb.Image(fig)}, commit=True)

    best_params = dict(best_nm_params, **other_params)
    return best_params, losses

# training function which only feeds context input to NM
def fit_mwg_context_nm_rnn(task_inputs, context_inputs, targets, loss_masks, params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                   plots=True, wandb_log=False, final_wandb_plot=False, orth_u=True, batch=False, batch_size=100, keyind=13): # added batch option
    opt_state = optimizer.init(params)
    N_data = task_inputs.shape[0]
    key = jr.PRNGKey(keyind)

    @jit
    def _step(params_and_opt, input):
        (params, opt_state, key) = params_and_opt
        if batch: 
            key, _ = jr.split(key, 2)
            batch_inds = jr.choice(key, jnp.arange(N_data), shape=(batch_size,))
            batch_context_inputs = context_inputs[batch_inds]
            batch_task_inputs = task_inputs[batch_inds]
            batch_targets = targets[batch_inds]
            batch_masks = loss_masks[batch_inds]
        else:
            batch_task_inputs = task_inputs
            batch_context_inputs = context_inputs
            batch_targets = targets
            batch_masks = loss_masks
        loss_value, grads = jax.value_and_grad(batched_context_nm_rnn_loss)(params, x0, z0, batch_task_inputs, batch_context_inputs, tau_x, tau_z, batch_targets, batch_masks, orth_u=orth_u)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, key), (params, loss_value)

    losses = []
    # sample_inputs, sample_targets, sample_masks = sample_one(jr.PRNGKey(1), T, intervals, measure_min, measure_max, delay, mask_pad)

    #sample_task_inputs, sample_context_inputs, sample_targets, sample_masks = task_inputs[0], context_inputs[0], targets[0], loss_masks[0] # grab a single trial to plot output

    best_loss = 1e6
    best_params = params
    for n in range(num_iters//1000):
        (params,_,_), (_, loss_values) = lax.scan(_step, (params, opt_state, key), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]

    return best_params, losses

# training function to fit only neuromodulatory parameters (context nm-rnn)
def fit_context_nm_only(task_inputs, context_inputs, targets, loss_masks, nm_params, other_params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                    wandb_log=False, orth_u=True, batch=False, batch_size=100, keyind=13): # training on full set of data
    opt_state = optimizer.init(nm_params)
    N_data = task_inputs.shape[0]
    key = jr.PRNGKey(keyind)

    @jit
    def _step(params_and_opt, input):
        (nm_params, opt_state, key) = params_and_opt
        if batch: 
            key, _ = jr.split(key, 2)
            batch_inds = jr.choice(key, jnp.arange(N_data), shape=(batch_size,))
            batch_context_inputs = context_inputs[batch_inds]
            batch_task_inputs = task_inputs[batch_inds]
            batch_targets = targets[batch_inds]
            batch_masks = loss_masks[batch_inds]
        else:
            batch_task_inputs = task_inputs
            batch_context_inputs = context_inputs
            batch_targets = targets
            batch_masks = loss_masks
        loss_value, grads = jax.value_and_grad(batched_context_nm_rnn_loss_frozen)(nm_params, other_params, x0, z0, batch_task_inputs, batch_context_inputs, tau_x, tau_z, batch_targets, batch_masks, orth_u=orth_u)
        updates, opt_state = optimizer.update(grads, opt_state, nm_params)
        nm_params = optax.apply_updates(nm_params, updates)
        return (nm_params, opt_state, key), (nm_params, loss_value)

    losses = []

    best_loss = 1e6
    params = dict(nm_params, **other_params)
    best_params = params
    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (nm_params,_,_), (_, loss_values) = lax.scan(_step, (nm_params, opt_state,key), None, length=1000) #arange bc the inputs aren't changing
        params = dict(nm_params, **other_params)
        losses.append(loss_values)
        losses.append(loss_values)
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]

    return best_params, losses

# training function to fit only input weights in low-rank RNN (comparison to only training NM in nm-rnn)
def fit_lr_inputweights_only(inputs, targets, loss_masks, input_params, other_params, optimizer, x0, num_iters, tau_x,
                    wandb_log=False, orth_u=True, batch=False, batch_size=100, keyind=13): # training on full set of data
    opt_state = optimizer.init(input_params)
    N_data = inputs.shape[0]
    key = jr.PRNGKey(keyind)

    @jit
    def _step(params_and_opt, input):
        (input_params, opt_state, key) = params_and_opt
        if batch: 
            key, _ = jr.split(key, 2)
            batch_inds = jr.choice(key, jnp.arange(N_data), shape=(batch_size,))
            batch_inputs = inputs[batch_inds]
            batch_targets = targets[batch_inds]
            batch_masks = loss_masks[batch_inds]
        else:
            batch_inputs = inputs
            batch_targets = targets
            batch_masks = loss_masks
        loss_value, grads = jax.value_and_grad(batched_lr_rnn_loss_split)(input_params, other_params, x0, batch_inputs, tau_x, batch_targets, batch_masks, orth_u=orth_u)
        updates, opt_state = optimizer.update(grads, opt_state, input_params)
        input_params = optax.apply_updates(input_params, updates)
        return (input_params, opt_state, key), (input_params, loss_value)

    losses = []

    best_loss = 1e6
    params = dict(input_params, **other_params)
    best_params = params
    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (input_params,_,_), (_, loss_values) = lax.scan(_step, (input_params, opt_state,key), None, length=1000) #arange bc the inputs aren't changing
        params = dict(input_params, **other_params)
        losses.append(loss_values)
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]

    return best_params, losses

# training function to fit only low-rank parameters 
def fit_mwg_lr_only(inputs, targets, loss_masks, nm_params, lr_params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                    plots=False, wandb_log=False, final_wandb_plot=False, orth_u=True): # training on full set of data
    opt_state = optimizer.init(lr_params)
    N_data = inputs.shape[0]

    @jit
    def _step(params_and_opt, input):
        (lr_params, opt_state) = params_and_opt
        #pdb.set_breakpoint()
        loss_value, grads = jax.value_and_grad(batched_nm_rnn_loss_frozen)(lr_params, nm_params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks)
        updates, opt_state = optimizer.update(grads, opt_state, lr_params)
        lr_params = optax.apply_updates(lr_params, updates)
        return (lr_params, opt_state), (lr_params, loss_value)

    losses = []
    sample_inputs, sample_targets, sample_masks = inputs[0], targets[0], loss_masks[0]  # grab a single trial to plot output
    
    best_loss = 1e6
    best_lr_params = lr_params

    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (lr_params,_), (_, loss_values) = lax.scan(_step, (lr_params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        # if n % 100 ç== 0:
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_lr_params = lr_params
            best_loss = loss_values[-1]

        params = dict(nm_params, **lr_params)
        ys, _, zs = nm_rnn(params, x0, z0, sample_inputs, tau_x, tau_z, orth_u=orth_u)

        if plots:
            # plt.figure(figsize=[10,6])
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=[10,6])
            # ax0.xlabel('Timestep')
            ax0.plot(sample_targets, label='True target')
            ax0.plot(ys, label='RNN target')
            ax0.legend()
            ax1.set_xlabel('Timestep')
            m = params['nm_sigmoid_weight']
            b = params['nm_sigmoid_intercept']
            ax1.plot(jax.nn.sigmoid((zs @ m.T + b)))
            # ax1.legend()
            plt.pause(0.1)

    if final_wandb_plot:
        # plt.figure(figsize=[10,6])
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=[10, 6])
        # ax0.xlabel('Timestep')
        ax0.set_title('after training nm only')
        ax0.plot(sample_targets, label='True target')
        ax0.plot(ys, label='RNN target')
        ax0.legend()
        ax1.set_xlabel('Timestep')
        m = params['nm_sigmoid_weight']
        b = params['nm_sigmoid_intercept']
        ax1.plot(jax.nn.sigmoid((zs @ m.T + b)))
        wandb.log({'nm_only_curves':wandb.Image(fig)}, commit=True)

    best_params = dict(nm_params, **best_lr_params)
    return best_params, losses

# training function for simplified case
def fit_lin_sym_nm_rnn(inputs, targets, loss_masks, params, optimizer, x0, z0, num_iters, tau_x, tau_z, wandb_log=False): # training on full set of data
    opt_state = optimizer.init(params)
    N_data = inputs.shape[0]

    @jit
    def _step(params_and_opt, input):
        (params, opt_state) = params_and_opt
        #pdb.set_breakpoint()
        loss_value, grads = jax.value_and_grad(batched_lin_sym_nm_rnn_loss)(params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []
    # sample_inputs, sample_targets, sample_masks = sample_one(jr.PRNGKey(1), T, intervals, measure_min, measure_max, delay, mask_pad)
    sample_inputs, sample_targets, sample_masks = inputs[0], targets[0], loss_masks[0] # grab a single trial to plot output

    fig = plt.figure(figsize=[10,6])
    best_loss = 1e6
    best_params = params
    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]
        # if n % 100 == 0:
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        ys, _, _ = lin_sym_nm_rnn(params, x0, z0, sample_inputs, tau_x, tau_z)

        fig.clf()
        plt.xlabel('Timestep')
        plt.plot(sample_targets, label='True target')
        plt.plot(ys, label='RNN target')
        plt.legend()
        plt.pause(0.1)

    return params, losses


def fit_lstm_mwg(inputs, targets, params, optimizer, init_carry, num_iters, wandb_log=False):
    opt_state = optimizer.init(params)
    (c0, h0) = init_carry

    @jit
    def _step(params_and_opt, input):
        (params, opt_state) = params_and_opt
        loss_value, grads = jax.value_and_grad(lstm_batched_loss)(params, c0, h0, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []

    best_loss = 1e6
    best_params = params
    for n in range(num_iters//1000):
        (params,_), (_,loss_values) = lax.scan(_step, (params, opt_state), None, length=1000)
        # (params,_), (_,loss_values) = _step(params, opt_state, init_carry, inputs, targets)
        losses.append(loss_values)
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]

    return best_params, losses

def fit_rnn_mwg(inputs, targets, loss_masks, params, optimizer, x0, num_iters, tau, 
                   wandb_log=False, orth_u=True, batch=False, batch_size=100, keyind=13): # training on full set of data
    opt_state = optimizer.init(params)
    N_data = inputs.shape[0]
    key = jr.PRNGKey(keyind)

    @jit
    def _step(params_and_opt, input):
        (params, opt_state, key) = params_and_opt
        if batch: 
            key, _ = jr.split(key, 2)
            batch_inds = jr.choice(key, jnp.arange(N_data), shape=(batch_size,))
            batch_inputs = inputs[batch_inds]
            batch_targets = targets[batch_inds]
            batch_masks = loss_masks[batch_inds]
        else:
            batch_inputs = inputs
            batch_targets = targets
            batch_masks = loss_masks
        loss_value, grads = jax.value_and_grad(batched_rnn_loss)(params, x0, batch_inputs, tau, batch_targets, batch_masks)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, key), (params, loss_value)

    losses = []
   
    best_loss = 1e6
    best_params = params
    for n in range(num_iters//1000):
        (params,_,_), (_, loss_values) = lax.scan(_step, (params, opt_state, key), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        if loss_values[-1] < best_loss: 
            best_params = params
            best_loss = loss_values[-1]
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})

    return best_params, losses