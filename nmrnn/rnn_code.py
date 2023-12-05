import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax
import matplotlib.pyplot as plt

# CODE FOR NM-RNN
def nm_rnn(params, x0, z0, inputs, tau_x, tau_z, nln=jnp.tanh):
    """
    Arguments:
    - params
    - x0
    - inputs
    - tau   : decay constant
    """

    U = params["row_factors"]       # D x R
    V = params["column_factors"]    # D x R
    B_xu = params["input_weights"]     # D x M
    C = params["readout_weights"]   # O x D
    W_zz = params["nm_rec_weight"]         # dim_nm x dim_nm
    B_zu = params["nm_input_weight"]      # dim_nm x M
    m = params["nm_sigmoid_weight"]         # scalar
    c = params["nm_sigmoid_intercept"]      # scalar

    N = x0.shape[0]

    def _step(x_and_z, u):
        x, z = x_and_z

        # update z
        z = (1.0 - (1. / tau_z)) * z + (1. / tau_z) * W_zz @ nln(z)
        z += (1. / tau_z) * B_zu @ u

        # update x
        s = jax.nn.sigmoid(m @ z + c)
        x = (1.0 - (1. / tau_x)) * x
        h = V.T @ nln(x)
        x += (1. / (tau_x * N)) * (U * s) @ h # divide by N
        x += (1. / tau_x) * B_xu @ u

        # calculate y
        y = C @ x
        return (x, z), (y, x, z)

    _, (ys, xs, zs) = lax.scan(_step, (x0, z0), inputs)

    return ys, xs, zs

batched_nm_rnn = vmap(nm_rnn, in_axes=(None, None, None, 0, None, None))

def batched_nm_rnn_loss(params, x0, z0, batch_inputs, tau_x, tau_z, batch_targets, batch_mask):
    ys, _, _ = batched_nm_rnn(params, x0, z0, batch_inputs, tau_x, tau_z)
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)


# CODE FOR LOW-RANK RNN
def lr_rnn(params, x0, inputs, tau, nln=jnp.tanh):
    """
    Arguments:
    - params
    - x0
    - inputs
    - tau   : decay constant
    """

    U = params["row_factors"]       # D x R
    V = params["column_factors"]    # D x R
    B = params["input_weights"]     # D x M
    C = params["readout_weights"]   # O x D

    N = x0.shape[0]

    def _step(x, u):
        h = V.T @ nln(x)
        x = (1.0 - (1. / tau)) * x + (1. / (tau * N)) * U @ h # divide by N
        x += (1. / tau) * B @ u
        y = C @ x
        return x, (y, x)

    _, (ys, xs) = lax.scan(_step, x0, inputs)

    return ys, xs

batched_lr_rnn = vmap(lr_rnn, in_axes=(None, None, 0, None))

def batched_lr_rnn_loss(params, x0, batch_inputs, tau, batch_targets, batch_mask):
    ys, _ = batched_lr_rnn(params, x0, batch_inputs, tau)
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)

def batched_nm_rnn_loss_frozen(nm_params, other_params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks):
    params = dict(nm_params, **other_params)
    return batched_nm_rnn_loss(params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks)