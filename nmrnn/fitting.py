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
from nmrnn.rnn_code import batched_nm_rnn_loss, nm_rnn, batched_lr_rnn_loss, lr_rnn, batched_nm_rnn_loss_frozen, lin_sym_nm_rnn, batched_lin_sym_nm_rnn_loss, context_nm_rnn, batched_context_nm_rnn_loss

def fit_mwg_nm_rnn(inputs, targets, loss_masks, params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                   plots=True, wandb_log=False, final_wandb_plot=False, orth_u=True): # training on full set of data
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
    # sample_inputs, sample_targets, sample_masks = sample_one(jr.PRNGKey(1), T, intervals, measure_min, measure_max, delay, mask_pad)

    sample_inputs, sample_targets, sample_masks = inputs[0], targets[0], loss_masks[0] # grab a single trial to plot output

    for n in range(num_iters//1000):
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})

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

    return params, losses


def fit_mwg_lr_rnn(inputs, targets, loss_masks, params, optimizer, x0, num_iters, tau, plots=False): # training on full set of data
    opt_state = optimizer.init(params)
    N_data = inputs.shape[0]

    @jit
    def _step(params_and_opt, input):
        (params, opt_state) = params_and_opt
        #pdb.set_breakpoint()
        loss_value, grads = jax.value_and_grad(batched_lr_rnn_loss)(params, x0, inputs, tau, targets, loss_masks)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []
    # sample_inputs, sample_targets, sample_masks = sample_one(jr.PRNGKey(1), T, intervals, measure_min, measure_max, delay, mask_pad)
    sample_inputs, sample_targets, sample_masks = inputs[0], targets[0], loss_masks[0] # grab a single trial to plot output

    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        # if n % 100 == 0:
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        ys, _ = lr_rnn(params, x0, sample_inputs, tau)

        if plots:
            plt.figure(figsize=[10,6])
            plt.xlabel('Timestep')
            plt.plot(sample_targets, label='True target')
            plt.plot(ys, label='RNN target')
            plt.legend()
            plt.pause(0.1)

    return params, losses


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
    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (nm_params,_), (_, loss_values) = lax.scan(_step, (nm_params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        # if n % 100 == 0:
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})

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

    return params, losses

# training function which only feeds context input to NM
def fit_mwg_context_nm_rnn(task_inputs, context_inputs, targets, loss_masks, params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                   plots=True, wandb_log=False, final_wandb_plot=False, orth_u=True): # training on full set of data
    opt_state = optimizer.init(params)
    N_data = task_inputs.shape[0]

    @jit
    def _step(params_and_opt, input):
        (params, opt_state) = params_and_opt
        loss_value, grads = jax.value_and_grad(batched_context_nm_rnn_loss)(params, x0, z0, task_inputs, context_inputs, tau_x, tau_z, targets, loss_masks, orth_u=orth_u)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []
    # sample_inputs, sample_targets, sample_masks = sample_one(jr.PRNGKey(1), T, intervals, measure_min, measure_max, delay, mask_pad)

    sample_task_inputs, sample_context_inputs, sample_targets, sample_masks = task_inputs[0], context_inputs[0], targets[0], loss_masks[0] # grab a single trial to plot output

    for n in range(num_iters//1000):
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})

        ys, _, zs = context_nm_rnn(params, x0, z0, sample_task_inputs, sample_context_inputs, tau_x, tau_z, orth_u=orth_u)

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

    return params, losses

# training function to fit only low-rank parameters 
def fit_mwg_lr_only(inputs, targets, loss_masks, nm_params, lr_params, optimizer, x0, z0, num_iters, tau_x, tau_z,
                    plots=False, wandb_log=False, final_wandb_plot=False): # training on full set of data
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
    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (lr_params,_), (_, loss_values) = lax.scan(_step, (lr_params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        # if n % 100 ç== 0:
        print(f'step {(n+1)*1000}, loss: {loss_values[-1]}')
        if wandb_log: wandb.log({'loss':loss_values[-1]})

        params = dict(nm_params, **lr_params)
        ys, _, zs = nm_rnn(params, x0, z0, sample_inputs, tau_x, tau_z)

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

    return params, losses

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
    for n in range(num_iters//1000):
        # (params, opt_state), loss_value = _step((params, opt_state))
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        if wandb_log: wandb.log({'loss':loss_values[-1]})
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