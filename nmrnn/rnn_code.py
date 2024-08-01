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
    rb = params["readout_bias"]             # O
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
        y = C @ x + rb
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
def lr_rnn(params, x0, inputs, tau, orth_u, nln=jnp.tanh):
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
    B = params["input_weights"]     # D x M
    C = params["readout_weights"]   # O x D
    rb = params["readout_bias"]             # O

    N = x0.shape[0]

    def _step(x, u):
        h = V.T @ nln(x)
        x = (1.0 - (1. / tau)) * x + (1. / (tau * N)) * U @ h # divide by N
        x += (1. / tau) * B @ u
        y = C @ x + rb
        return x, (y, x)

    _, (ys, xs) = lax.scan(_step, x0, inputs)

    return ys, xs

batched_lr_rnn = vmap(lr_rnn, in_axes=(None, None, 0, None, None))

def batched_lr_rnn_loss(params, x0, batch_inputs, tau, batch_targets, batch_mask, orth_u=True):
    ys, _ = batched_lr_rnn(params, x0, batch_inputs, tau, orth_u)
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)

# for training on only subset of params
def batched_lr_rnn_loss_split(input_params, other_params, x0, inputs, tau_x, targets, loss_masks, orth_u=True):
    params = dict(input_params, **other_params)
    return batched_lr_rnn_loss(params, x0, inputs, tau_x, targets, loss_masks, orth_u=orth_u)

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
    rb = params["readout_bias"]             # O
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
    rb = params["readout_bias"]             # O
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
        y = C @ x + rb
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

def simple_lstm(params, c0, h0, inputs, gate_fn=jax.nn.sigmoid, activation_fn=jnp.tanh):
    """
    Arguments:
    - params: dict of LSTM weights and cell states
    - c0: initial cell (memory) state
    - h0: initial hidden state
    - inputs: sequence of inputs passed in, with batch dims first
    """
    # N -- hidden size, M -- input_size, O -- output_size

    # (c0, h0) = init_carry

    N = h0.shape[0] # hidden size of LSTM

    W_iu = params['weights_iu']     # N x M
    W_ih = params['weights_ih']     # N x N
    b_ih = params['bias_ih']        # scalar

    W_fu = params['weights_fu']     # N x M
    W_fh = params['weights_fh']     # N x N
    b_fh = params['bias_fh']        # scalar

    W_gu = params['weights_gu']     # N x M
    W_gh = params['weights_gh']     # N x N
    b_gh = params['bias_gh']        # scalar

    W_ou = params['weights_ou']     # N x M
    W_oh = params['weights_oh']     # N x N
    b_oh = params['bias_oh']        # scalar

    C = params["readout_weights"]   # O x N

    def _step(carry, u):
        c, h = carry
        i = gate_fn(W_iu @ u + W_ih @ h + b_ih)
        f = gate_fn(W_fu @ u + W_fh @ h + b_fh)
        g = activation_fn(W_gu @ u + W_gh @ h + b_gh)
        o = gate_fn(W_ou @ u + W_oh @ h + b_oh)

        new_c = f * c + i * g
        new_h = o * activation_fn(new_c)

        y = C @ new_h

        return (new_c, new_h), y

    if 'embedding_weights' in params:
      embedding = params['embedding_weights']
      inputs = embedding[inputs]    # make input batch_size x max_len x num_features in the case of text sentiment analysis

    carry, y = lax.scan(_step, (c0, h0), inputs)
    return carry, y

def initialize_carry(hidden_size, key):
    key1, key2 = jr.split(key)
    mem_shape = (hidden_size, )
    scale_factor = 1.0 / jnp.sqrt(hidden_size)

    c = jr.normal(key1, mem_shape) * scale_factor
    h = jr.normal(key2, mem_shape) * scale_factor
    carry = (c, h)
    return carry

# should initial carry also be batched?
batched_simple_lstm = vmap(simple_lstm, in_axes=(None, None, None, 0))

def lstm_batched_loss(params, c0, h0, batch_inputs, batch_targets):
    _, ys = batched_simple_lstm(params, c0, h0, batch_inputs)
    return jnp.mean(((ys - batch_targets)**2))

def initialize_carry(hidden_size, key):
    key1, key2 = jr.split(key)
    mem_shape = (hidden_size, )
    scale_factor = 1.0 / jnp.sqrt(hidden_size)

    c = jr.normal(key1, mem_shape) * scale_factor
    h = jr.normal(key2, mem_shape) * scale_factor
    carry = (c, h)
    return carry