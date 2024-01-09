import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
from jax.tree_util import tree_map
import optax
import numpy as np

# import pickle as pkl

# import matplotlib.pyplot as plt
# import pdb

# from flax import linen as nn
import functools
import random
import math

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

def generate_sequences_and_inds(batch_size, seq_len, key, seq_bound=10):
    key, skey, sskey = jr.split(key, 3)
    sequences = jr.randint(skey, (batch_size, seq_len), -seq_bound, seq_bound+1)
    inds = jr.randint(sskey, (batch_size, 1), 0, seq_len)
    inputs = jnp.concatenate([inds, sequences], axis=1)

    outputs = jnp.choose(inputs[:, 0], inputs[:, 1:].T)

    return jnp.expand_dims(inputs, axis=-1), outputs


def generate_tensor_sequences_and_inds(batch_size, seq_len, seq_bound=10):
    sequences = torch.randint(-10, 10, (batch_size, seq_len))
    inds = torch.randint(0, seq_len, (batch_size,))
    inputs = torch.cat((inds.unsqueeze(-1), sequences), axis=1)

    outputs = sequences[torch.arange(len(inds)), inds]

    return inputs.unsqueeze(-1), outputs


def generate_sequences_and_fixed_query(batch_size, idx, seq_len, key, seq_bound=10, fixed_value=None):
    assert idx >= 0 and idx < seq_len
    key, skey, sskey = jr.split(key, 3)
    sequences = jr.randint(skey, (batch_size, seq_len), -seq_bound, seq_bound+1)
    idx_arr = jnp.repeat(idx, batch_size).reshape((-1, 1))

    if fixed_value is not None:
        value_col = jnp.repeat(fixed_value, batch_size)
        sequences = sequences.at[:, idx].set(value_col)
        # print(sequences[:, idx])

    inputs = jnp.concatenate([idx_arr, sequences], axis=1)

    outputs = jnp.choose(inputs[:, 0], inputs[:, 1:].T)

    return jnp.expand_dims(inputs, axis=-1), outputs
