import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax
import matplotlib.pyplot as plt

from data_generation import sample_one
from rnn_code import batched_nm_rnn_loss, nm_rnn

def fit_mwg_nm_rnn(inputs, targets, loss_masks, params, optimizer, x0, z0, num_iters, tau_x, tau_z): # training on full set of data
    opt_state = optimizer.init(params)
    N_data = inputs.shape[0]

    @jit
    def _step(params_and_opt, input):
        (params, opt_state) = params_and_opt
        loss_value, grads = jax.value_and_grad(batched_nm_rnn_loss)(params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []
    # sample_inputs, sample_targets, sample_masks = sample_one(jr.PRNGKey(1), T, intervals, measure_min, measure_max, delay, mask_pad)

    sample_inputs, sample_targets, sample_masks = inputs[0], targets[0], loss_masks[0] # grab a single trial to plot output

    for n in range(num_iters//1000):
        (params,_), (_, loss_values) = lax.scan(_step, (params, opt_state), None, length=1000) #arange bc the inputs aren't changing
        losses.append(loss_values)
        print(f'step {n*1000}, loss: {loss_values[-1]}')
        ys, _, zs = nm_rnn(params, x0, z0, sample_inputs, tau_x, tau_z)

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

    return params, losses