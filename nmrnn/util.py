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

def percent_correct(inputs, target_outputs, model_outputs):
    num_samples = inputs.shape[0]
    correct_count = 0
    def _trial_correct(sample_in, sample_out, model_out):
        assert sample_out.shape == model_out.shape, "sample_out and model_out are not same shape!"
        time_fix_off = jnp.argmin(sample_in[0])
        if (model_out[0, :time_fix_off] > 0.5).all():
            dist = decode_angle(sample_out[1,-1],sample_out[2,-1]) - decode_angle(model_out[1, -1], model_out[2,-1])
            diff = jnp.minimum(abs(dist), 2*jnp.pi-abs(dist))
            if diff < jnp.pi/10:
                return 1
            else: return 0
        else: return 0
    for i in range(num_samples):
        correct_count += _trial_correct(inputs[i], target_outputs[i], model_outputs[i].T)
    return correct_count/num_samples

def decode_angle(sin_angle, cos_angle):
    # decodes angle from sine/cosine readout, ignoring scaling
    return jnp.arctan(sin_angle/cos_angle)

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
    skeys = jr.split(key, 9)
    #   hscale = 0.1
    ifactor = 1.0 / jnp.sqrt(u) # scaling of input weights
    # hfactor = g / jnp.sqrt(n) # scaling of recurrent weights
    pfactor = 1.0 / jnp.sqrt(n) # scaling of output weights

    row_factors = jnp.zeros((n,r))
    column_factors = jnp.zeros((n,r))

    row_factors = jr.normal(skeys[0],(n,r))
    column_factors = jr.normal(skeys[1],(n,r))

    return {'row_factors' : row_factors,
            'column_factors' : column_factors,
            'input_weights' : jr.normal(skeys[2], (n,u))*ifactor,
            'readout_weights' : jr.normal(skeys[3], (o,n))*pfactor,
            'readout_bias' : jr.normal(skeys[8], (o,))*pfactor,
            'nm_rec_weight' : jr.normal(skeys[4], (m,m))*0.1,
            'nm_input_weight' : jr.normal(skeys[5], (m,u))*ifactor,
            'nm_sigmoid_weight' : jr.normal(skeys[6], (k,m))*0.1,
            'nm_sigmoid_intercept' : jr.normal(skeys[7], (k,))*0.1}

def random_lrrnn_params(key, u, n, r, o, g=1.0):
    """Generate random low-rank RNN parameters

    Arguments:
    u:  number of inputs
    n:  number of neurons in main network
    r:  rank of main network
    o:  number of outputs
    """
    skeys = jr.split(key, 5)
    #   hscale = 0.1
    ifactor = 1.0 / jnp.sqrt(u) # scaling of input weights
    # hfactor = g / jnp.sqrt(n) # scaling of recurrent weights
    pfactor = 1.0 / jnp.sqrt(n) # scaling of output weights

    row_factors = jnp.zeros((n,r))
    column_factors = jnp.zeros((n,r))

    row_factors = jr.normal(skeys[0],(n,r))
    column_factors = jr.normal(skeys[1],(n,r))

    return {'row_factors' : row_factors,
            'column_factors' : column_factors,
            'input_weights' : jr.normal(skeys[2], (n,u))*ifactor,
            'readout_weights' : jr.normal(skeys[3], (o,n))*pfactor,
            'readout_bias' : jr.normal(skeys[8], (o,))*pfactor}


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

def load_wandb_model(filepath):
    artifact = wandb.use_artifact(filepath, type="model")
    artifact_dir = artifact.download()
    model_infile = open(os.path.join(artifact_dir, "model"), 'rb')
    model = pickle.load(model_infile)
    model_infile.close()
    return model