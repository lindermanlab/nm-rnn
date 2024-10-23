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


############### CODE FOR LSTM
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

def batched_lstm_loss_split(input_params, other_params, c0, h0, inputs, targets):
    params = dict(input_params, **other_params)
    return lstm_batched_loss(params, c0, h0, inputs, targets)

########### CODE FOR VANILLA RNN
def rnn(params, x0, inputs, tau, nln=jnp.tanh):
    """
    Arguments:
    - params
    - x0
    - inputs
    - tau   : decay constant
    """

    W = params["recurrent_weights"]       # D x D
    B = params["input_weights"]     # D x M
    C = params["readout_weights"]   # O x D
    rb = params["readout_bias"]             # O

    N = x0.shape[0]

    def _step(x, u):
        h = nln(x)
        x = (1.0 - (1. / tau)) * x + (1. / (tau * N)) * W @ h # divide by N
        x += (1. / tau) * B @ u
        y = C @ x + rb
        return x, (y, x)

    _, (ys, xs) = lax.scan(_step, x0, inputs)

    return ys, xs

batched_rnn = vmap(rnn, in_axes=(None, None, 0, None))

def batched_rnn_loss(params, x0, batch_inputs, tau, batch_targets, batch_mask):
    ys, _ = batched_rnn(params, x0,batch_inputs, tau)
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)

# for training on only subset of params
def batched_rnn_loss_split(input_params, other_params, x0, inputs, tau_x, targets, loss_masks):
    params = dict(input_params, **other_params)
    return batched_rnn_loss(params, x0, inputs, tau_x, targets, loss_masks)


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

batched_multiregion_nm_rnn = vmap(multiregion_nmrnn, in_axes=(None, None, None, 0, None, None, None))

def batched_multiregion_nm_rnn_loss(params, x0, z0, batch_inputs, tau_x, tau_z, batch_targets, batch_mask, modulation=True):
    ys, _, _ = batched_nm_rnn(params, x0, z0, batch_inputs, tau_x, tau_z, modulation) # removed orth_u from here
    return jnp.sum(((ys - batch_targets)**2)*batch_mask)/jnp.sum(batch_mask)