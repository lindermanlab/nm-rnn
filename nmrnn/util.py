import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax
import wandb
import os
import pickle

import matplotlib.pyplot as plt

def random_nmrnn_params(key, u, n, r, m, k, o, g=1.0):
    """Generate random low-rank RNN parameters

    Arguments:
    u:  number of inputs
    n:  number of neurons in main network
    r:  rank of main network
    m:  number of neurons in NM network
    k:  dimension of NM input affecting weight matrix (either 1 or r)
    o:  number of outputs
    """
    skeys = jr.split(key, 8)
    #   hscale = 0.1
    ifactor = 1.0 / jnp.sqrt(u) # scaling of input weights
    # hfactor = g / jnp.sqrt(n) # scaling of recurrent weights
    pfactor = 1.0 / jnp.sqrt(n) # scaling of output weights

    row_factors = jnp.zeros((n,r))
    column_factors = jnp.zeros((n,r))

    for i in range(r):
        sample = jr.multivariate_normal(key, jnp.zeros((2,)), jnp.array([[1.,0.8],[0.8,1.]]), shape=(n,))
        # pdb.set_trace()
        row_factors = row_factors.at[:,i].set(sample[:,0])
        column_factors = column_factors.at[:,i].set(sample[:,1])

    return {'row_factors' : row_factors,
            'column_factors' : column_factors,
            'input_weights' : jr.normal(skeys[2], (n,u))*ifactor,
            'readout_weights' : jr.normal(skeys[3], (o,n))*pfactor,
            'nm_rec_weight' : jr.normal(skeys[4], (m,m))*0.1,
            'nm_input_weight' : jr.normal(skeys[5], (m,u))*ifactor,
            'nm_sigmoid_weight' : jr.normal(skeys[6], (k,m))*0.1,
            'nm_sigmoid_intercept' : jr.normal(skeys[7], (k,))*0.1}


def log_wandb_model(model, name, type):
    trained_model_artifact = wandb.Artifact(name,type=type)
    if not os.path.isdir('models'): os.mkdir('models')
    subdirectory = wandb.run.name
    filepath = os.path.join('models', subdirectory)
    os.mkdir(filepath)
    obs_outfile = open(os.path.join(filepath, "model"), 'wb')
    pickle.dump(model, obs_outfile)
    obs_outfile.close()
    trained_model_artifact.add_dir(filepath)
    wandb.log_artifact(trained_model_artifact)