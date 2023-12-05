import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, vmap, jit
from jax import lax
import optax

import matplotlib.pyplot as plt

def sample_one(key,
               T,
               intervals,
               measure_min,
               measure_max,
               delay,
               mask_pad=None,
               output_min=-0.5,
               output_max=0.5):
    """
    Simulates inputs for the measure-set-go task.

    Arguments:
    T: length of the trial
    intervals: (num_contexts, num_intervals) array of intervals for each contet
    measure_min: minimum time when measure cue comes
    measure_max: maximum time when measure cue comes
    delay: length of time between measure and go
    mask_pad: # indices that loss masks extends to on either side of ramp
        if None, mask is all ones (no masking loss)

    Returns:
    inputs: (T, 3) array of time series (measure, set, go) where
            measure, set, go are one-hot time series marking each event

    output: (T, 1) ramp starting at go and lasting (set - measure) time

    mask: (T, 1) binary mask over ramp +- mask_pad indices

    Note: outputs are aligned at *go* cue
    """
    num_contexts, num_intervals = intervals.shape

    k1, k2, k3 = jr.split(key, 3)
    t_measure = jr.randint(k1, (), measure_min, measure_max)
    context = jr.choice(k2, num_contexts)
    interval = jr.choice(k3, intervals[context])
    t_set = t_measure + interval
    t_go = t_set + delay

    # construct the input array
    inputs = jnp.column_stack([
        10*(jnp.arange(T) == t_measure),
        10*(jnp.arange(T) == t_set),
        10*(jnp.arange(T) == t_go),
        # context * jnp.ones(T)
    ])

    # construct the outputs
    slope = 1.0 / interval
    intercept = -t_go * slope
    outputs = jnp.clip(intercept + jnp.arange(T) * slope, 0, 1)
    outputs = output_min + (output_max - output_min) * outputs

    max_interval = jnp.max(intervals)
    if mask_pad is not None: mask = (jnp.arange(T) >= t_go-mask_pad) & (jnp.arange(T) < t_go + max_interval + mask_pad)
    else: mask = jnp.ones(T)

    return inputs, outputs[:, None], mask[:, None]

def sample_all(T,
               intervals,
               measure_min,
               measure_max,
               delay,
               mask_pad=None,
               output_min=-0.5,
               output_max=0.5):
    """
    Simulates all possible input/output pairs for the measure-set-go task.

    Arguments:
    T: length of the trial
    intervals: (num_contexts, num_intervals) array of intervals for each contet
    measure_min: minimum time when measure cue comes
    measure_max: maximum time when measure cue comes
    delay: length of time between measure and go

    Returns:
    inputs: (T, 3) array of time series (measure, set, go, tonic cue) where
            measure, set, go are one-hot time series marking each event

    output: (T, 1) ramp starting at go and lasting (set - measure) time

    mask: (T, 1) binary mask over ramp +- 300 units of time

    Note: outputs are aligned at *go* cue
    """
    num_contexts, num_intervals = intervals.shape

    def _single(context_ind, interval_ind, t_measure):
        # construct the input array
        choice_intervals = intervals[context_ind][0]
        max_interval = jnp.max(intervals)
        this_interval = choice_intervals[interval_ind]
        t_set = t_measure + this_interval
        t_go = t_set + delay
        inputs = jnp.column_stack([
            10*(jnp.arange(T) == t_measure),
            10*(jnp.arange(T) == t_set),
            10*(jnp.arange(T) == t_go),
            # context_ind * jnp.ones(T)
        ])

        # construct the outputs
        slope = 1.0 / this_interval
        intercept = -t_go * slope
        outputs = jnp.clip(intercept + jnp.arange(T) * slope, 0, 1)
        outputs = output_min + (output_max - output_min) * outputs

        # construct the mask
        if mask_pad is not None: mask = (jnp.arange(T) >= t_go-mask_pad) & (jnp.arange(T) < t_go + max_interval + mask_pad)
        else: mask = jnp.ones(T)

        return inputs, outputs, mask

    context_inds = jnp.arange(num_contexts)
    interval_inds = jnp.arange(num_intervals)
    possible_t_measure = jnp.arange(measure_min, measure_max)

    batch_contexts = jnp.repeat(context_inds, num_intervals*(measure_max-measure_min))[:,None]
    batch_intervals = jnp.repeat(interval_inds, num_contexts*(measure_max-measure_min))[:,None]
    batch_t_measures = jnp.tile(possible_t_measure, num_contexts*num_intervals)[:,None]

    all_inputs, all_outputs, all_masks, = jax.vmap(_single)(batch_contexts, batch_intervals, batch_t_measures)

    return all_inputs, all_outputs[:,:,None], all_masks[:,:,None]