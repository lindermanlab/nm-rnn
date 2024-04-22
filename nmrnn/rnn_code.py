import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax
import matplotlib.pyplot as plt

# CODE FOR NM-RNN
def nm_rnn(params, x0, z0, inputs, tau_x, tau_z, orth_u, nln=jnp.tanh):
    """
    Arguments:
    - params
    - x0
    - inputs
    - tau   : decay constant
    """

    U = params["row_factors"]       # D x R
    if orth_u: U, _ = jnp.linalg.qr(U)         # orthogonalize
    V = params["column_factors"]    # D x R
    # V, _ = jnp.linalg.qr(V)         # orthogonalize
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
        xp = x # hold onto previous value
        s = jax.nn.sigmoid(m @ z + c) # calculate nm signal
        x = (1.0 - (1. / tau_x)) * xp # decay term
        h = V.T @ nln(xp)
        #x += (1. / (tau_x * N)) * (U * s) @ h  # divide by N
        x += (1. / (tau_x)) * (U * s) @ h # don't divide by N now because we normalized U
        x += (1. / tau_x) * B_xu @ u

        # calculate y
        y = C @ x
        return (x, z), (y, x, z)

    _, (ys, xs, zs) = lax.scan(_step, (x0, z0), inputs)

    return ys, xs, zs

batched_nm_rnn = vmap(nm_rnn, in_axes=(None, None, None, 0, None, None, None))

def batched_nm_rnn_loss(params, x0, z0, batch_inputs, tau_x, tau_z, batch_targets, batch_mask, orth_u=True):
    ys, _, _ = batched_nm_rnn(params, x0, z0, batch_inputs, tau_x, tau_z, orth_u)
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)

# don't compute loss over LR params
def batched_nm_rnn_loss_frozen(nm_params, other_params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks, orth_u=True):
    params = dict(nm_params, **other_params)
    return batched_nm_rnn_loss(params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks, orth_u=orth_u)

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

# CODE FOR LINEAR SYMMETRIC NM-RNN
def lin_sym_nm_rnn(params, x0, z0, inputs, tau_x, tau_z):
    """
    Arguments:
    - params
    - x0
    - inputs
    - tau   : decay constant
    """

    U = params["row_factors"]       # D x R
    U, _ = jnp.linalg.qr(U) # orthogonalize
    V = U    # D x R
    B_xu = params["input_weights"]     # D x M
    C = params["readout_weights"]   # O x D
    try: rb = params["readout_bias"]             # O
    except: rb = jnp.zeros(C.shape[0])
    W_zz = params["nm_rec_weight"]         # dim_nm x dim_nm
    B_zu = params["nm_input_weight"]      # dim_nm x M
    m = params["nm_sigmoid_weight"]         # scalar
    c = params["nm_sigmoid_intercept"]      # scalar

    N = x0.shape[0]

    def _step(x_and_z, u):
        x, z = x_and_z

        # update z
        z = (1.0 - (1. / tau_z)) * z + (1. / tau_z) * W_zz @ jnp.tanh(z) # remove this?
        z += (1. / tau_z) * B_zu @ u

        # update x
        xp = x # hold onto previous value
        s = jax.nn.sigmoid(m @ z + c) # calculate nm signal
        x = (1.0 - (1. / tau_x)) * xp # decay term
        h = V.T @ xp
        x += (1. / tau_x) * (U * s) @ h  # divide by N
        x += (1. / tau_x) * B_xu @ u

        # calculate y
        y = C @ x + rb
        return (x, z), (y, x, z)

    _, (ys, xs, zs) = lax.scan(_step, (x0, z0), inputs)

    return ys, xs, zs

batched_lin_sym_nm_rnn = vmap(lin_sym_nm_rnn, in_axes=(None, None, None, 0, None, None))

def batched_lin_sym_nm_rnn_loss(params, x0, z0, batch_inputs, tau_x, tau_z, batch_targets, batch_mask):
    ys, _, _ = batched_lin_sym_nm_rnn(params, x0, z0, batch_inputs, tau_x, tau_z)
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)

# CODE FOR MULTITASK NM-RNN WHERE ONLY NM RECEIVES CONTEXT
def context_nm_rnn(params, x0, z0, task_inputs, context_inputs, tau_x, tau_z, orth_u, nln=jnp.tanh):
    """
    Arguments:
    - params
    - x0
    - task_inputs
    - context_inputs
    - tau   : decay constant
    """

    U = params["row_factors"]       # D x R
    if orth_u: U, _ = jnp.linalg.qr(U)         # orthogonalize
    V = params["column_factors"]    # D x R
    # V, _ = jnp.linalg.qr(V)         # orthogonalize
    B_xu = params["input_weights"]     # D x M
    C = params["readout_weights"]   # O x D
    W_zz = params["nm_rec_weight"]         # dim_nm x dim_nm
    B_zu = params["nm_input_weight"]      # dim_nm x (M + dim_context)
    m = params["nm_sigmoid_weight"]         # scalar
    c = params["nm_sigmoid_intercept"]      # scalar

    N = x0.shape[0]

    inputs_x = task_inputs # T x M
    #TODO: check concat
    inputs_z = jnp.concatenate((task_inputs, context_inputs), axis=-1) # T x (M + dim_context)

    def _step(x_and_z, u):
        x, z = x_and_z
        u_x, u_z = u

        # update z
        z = (1.0 - (1. / tau_z)) * z + (1. / tau_z) * W_zz @ nln(z)
        z += (1. / tau_z) * B_zu @ u_z

        # update x
        xp = x # hold onto previous value
        s = jax.nn.sigmoid(m @ z + c) # calculate nm signal
        x = (1.0 - (1. / tau_x)) * xp # decay term
        h = V.T @ nln(xp)
        #x += (1. / (tau_x * N)) * (U * s) @ h  # divide by N
        x += (1. / (tau_x)) * (U * s) @ h # don't divide by N now because we normalized U
        x += (1. / tau_x) * B_xu @ u_x

        # calculate y
        y = C @ x
        return (x, z), (y, x, z)

    _, (ys, xs, zs) = lax.scan(_step, (x0, z0), (inputs_x, inputs_z))

    return ys, xs, zs

batched_context_nm_rnn = vmap(context_nm_rnn, in_axes=(None, None, None, 0, 0, None, None, None))

def batched_context_nm_rnn_loss(params, x0, z0, batch_task_inputs, batch_context_inputs, tau_x, tau_z, batch_targets, batch_mask, orth_u=True):
    ys, _, _ = batched_context_nm_rnn(params, x0, z0, batch_task_inputs, batch_context_inputs, tau_x, tau_z, orth_u)
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)

# for training on only nm or lr params
def batched_context_nm_rnn_loss_frozen(nm_params, other_params, x0, z0, task_inputs, context_inputs, tau_x, tau_z, targets, loss_masks, orth_u=True):
    params = dict(nm_params, **other_params)
    return batched_context_nm_rnn_loss(params, x0, z0, task_inputs, context_inputs, tau_x, tau_z, targets, loss_masks, orth_u=orth_u)