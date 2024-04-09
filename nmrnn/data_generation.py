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


### DUNCKER/DRISCOLL 2020 TASKS


def sample_delay_pro(key,
                     T):
    # generate theta
    key_stim, key_trialsplit,  = jr.split(key, 2)
    theta_in = jr.uniform(key_stim, minval=0., maxval=2*jnp.pi)
    theta_out = theta_in # values are the for this task

    # segment trial into different periods
    key_stim_on, key_fix_off, key_response = jr.split(key_trialsplit,3)
    t_stim_on = jax.random.uniform(key_stim_on, minval=0.3, maxval=0.7) # time between 0 and when the stimulus comes on
    t_fix_off = t_stim_on + jax.random.uniform(key_fix_off, minval=0.2, maxval=1.5) # time between when the stimulus comes on and when the go cue arrives (fix-off)
    t_response = t_fix_off + jax.random.uniform(key_response, minval=0.3, maxval=0.7) # time between go cue and end of trial

    t_total = t_response # total sampled time of trial

    t_stim_on = t_stim_on/t_total * T
    t_fix_off = t_fix_off/t_total * T

    t = jnp.arange(T)
    # make fixation input
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [1., 0.]
    fix_input = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make stim inputs
    stim_scale = 1.
    cond_list = [t<t_stim_on, t>=t_stim_on]
    func_list_1 = [0, stim_scale * jnp.sin(theta_in)]
    func_list_2 = [0, stim_scale * jnp.cos(theta_in)]
    stim_input_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1]
    stim_input_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1]
    stim_input = jnp.array([stim_input_1, stim_input_2])

    # make fixation output
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [0.8, 0]
    fix_output = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make response outputs
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list_1 = [0, jnp.sin(theta_out)]
    func_list_2 = [0, jnp.cos(theta_out)]
    response_output_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1]
    response_output_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1]
    response_output = jnp.array([response_output_1, response_output_2])

    return jnp.vstack((fix_input, stim_input)), jnp.vstack((fix_output, response_output))


def sample_delay_anti(key,
                     T):
    # generate theta
    key_stim, key_trialsplit,  = jr.split(key, 2)
    theta_in = jr.uniform(key_stim, minval=0., maxval=2*jnp.pi)
    theta_out = theta_in + jnp.pi # opposite direction

    # segment trial into different periods
    key_stim_on, key_fix_off, key_response = jr.split(key_trialsplit,3)
    t_stim_on = jax.random.uniform(key_stim_on, minval=0.3, maxval=0.7) # time between 0 and when the stimulus comes on
    t_fix_off = t_stim_on + jax.random.uniform(key_fix_off, minval=0.2, maxval=1.5) # time between when the stimulus comes on and when the go cue arrives (fix-off)
    t_response = t_fix_off + jax.random.uniform(key_response, minval=0.3, maxval=0.7) # time between go cue and end of trial

    t_total = t_response # total sampled time of trial

    t_stim_on = t_stim_on/t_total * T
    t_fix_off = t_fix_off/t_total * T

    t = jnp.arange(T)
    # make fixation input
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [1., 0.]
    fix_input = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make stim inputs
    stim_scale = 1.
    cond_list = [t<t_stim_on, t>=t_stim_on]
    func_list_1 = [0, stim_scale * jnp.sin(theta_in)]
    func_list_2 = [0, stim_scale * jnp.cos(theta_in)]
    stim_input_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1]
    stim_input_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1]
    stim_input = jnp.array([stim_input_1, stim_input_2])

    # make fixation output
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [0.8, 0]
    fix_output = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make response outputs
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list_1 = [0, jnp.sin(theta_out)]
    func_list_2 = [0, jnp.cos(theta_out)]
    response_output_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1]
    response_output_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1]
    response_output = jnp.array([response_output_1, response_output_2])

    return jnp.vstack((fix_input, stim_input)), jnp.vstack((fix_output, response_output))


def sample_memory_pro(key,
                     T):
    # generate theta
    key_stim, key_trialsplit,  = jr.split(key, 2)
    theta_in = jr.uniform(key_stim, minval=0., maxval=2*jnp.pi)
    theta_out = theta_in # values are the for this task

    # segment trial into different periods
    key_stim_on, key_stim_off, key_fix_off, key_response = jr.split(key_trialsplit,4)
    t_stim_on = jax.random.uniform(key_stim_on, minval=0.3, maxval=0.7) # time between 0 and when the stimulus comes on
    t_stim_off = t_stim_on + jax.random.uniform(key_stim_off, minval=0.2, maxval=1.6) # time between when the stimulus comes on and goes off
    t_fix_off = t_stim_off + jax.random.uniform(key_fix_off, minval=0.2, maxval=1.6) # time between when the stimulus comes on and when the go cue arrives (fix-off)
    t_response = t_fix_off + jax.random.uniform(key_response, minval=0.3, maxval=0.7) # time between go cue and end of trial

    t_total = t_response # total sampled time of trial

    t_stim_on = t_stim_on/t_total * T
    t_stim_off = t_stim_off/t_total * T
    t_fix_off = t_fix_off/t_total * T

    t = jnp.arange(T)
    # make fixation input
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [1., 0.]
    fix_input = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make stim inputs
    stim_scale = 1.
    cond_list = [t<t_stim_on, jnp.logical_and(t>=t_stim_on, t<t_stim_off), t >=t_stim_off]
    func_list_1 = [0, stim_scale * jnp.sin(theta_in), 0]
    func_list_2 = [0, stim_scale * jnp.cos(theta_in), 0]
    stim_input_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1] + cond_list[2]*func_list_1[2]
    stim_input_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1] + cond_list[2]*func_list_2[2]
    stim_input = jnp.array([stim_input_1, stim_input_2])

    # make fixation output
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [0.8, 0]
    fix_output = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make response outputs
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list_1 = [0, jnp.sin(theta_out)]
    func_list_2 = [0, jnp.cos(theta_out)]
    response_output_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1]
    response_output_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1]
    response_output = jnp.array([response_output_1, response_output_2])

    return jnp.vstack((fix_input, stim_input)), jnp.vstack((fix_output, response_output))


def sample_memory_anti(key,
                     T):
    # generate theta
    key_stim, key_trialsplit,  = jr.split(key, 2)
    theta_in = jr.uniform(key_stim, minval=0., maxval=2*jnp.pi)
    theta_out = theta_in +  jnp.pi # opposite

    # segment trial into different periods
    key_stim_on, key_stim_off, key_fix_off, key_response = jr.split(key_trialsplit,4)
    t_stim_on = jax.random.uniform(key_stim_on, minval=0.3, maxval=0.7) # time between 0 and when the stimulus comes on
    t_stim_off = t_stim_on + jax.random.uniform(key_stim_off, minval=0.2, maxval=1.6) # time between when the stimulus comes on and goes off
    t_fix_off = t_stim_off + jax.random.uniform(key_fix_off, minval=0.2, maxval=1.6) # time between when the stimulus comes on and when the go cue arrives (fix-off)
    t_response = t_fix_off + jax.random.uniform(key_response, minval=0.3, maxval=0.7) # time between go cue and end of trial

    t_total = t_response # total sampled time of trial

    t_stim_on = t_stim_on/t_total * T
    t_stim_off = t_stim_off/t_total * T
    t_fix_off = t_fix_off/t_total * T

    t = jnp.arange(T)
    # make fixation input
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [1., 0.]
    fix_input = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make stim inputs
    stim_scale = 1.
    cond_list = [t<t_stim_on, jnp.logical_and(t>=t_stim_on, t<t_stim_off), t >=t_stim_off]
    func_list_1 = [0, stim_scale * jnp.sin(theta_in), 0]
    func_list_2 = [0, stim_scale * jnp.cos(theta_in), 0]
    stim_input_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1] + cond_list[2]*func_list_1[2]
    stim_input_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1] + cond_list[2]*func_list_2[2]
    stim_input = jnp.array([stim_input_1, stim_input_2])

    # make fixation output
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [0.8, 0]
    fix_output = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make response outputs
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list_1 = [0, jnp.sin(theta_out)]
    func_list_2 = [0, jnp.cos(theta_out)]
    response_output_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1]
    response_output_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1]
    response_output = jnp.array([response_output_1, response_output_2])

    return jnp.vstack((fix_input, stim_input)), jnp.vstack((fix_output, response_output))

def random_trials(key, task_list, T, num_trials, fix_output=False):
    key1, key2 = jr.split(key, 2)
    num_tasks = len(task_list)
    random_order = jr.choice(key1, num_tasks, shape=(num_trials,))
   
    sample_keys_trials = jr.split(key2, num_trials)

    samples_in = jnp.zeros((num_trials, 3 + num_tasks, T))
    samples_out = jnp.zeros((num_trials, 3, T))
    for i in range(num_trials):
        sample_in, sample_out = task_list[random_order[i]](sample_keys_trials[i], T)
        samples_in = samples_in.at[i, :3, :].set(sample_in)
        samples_out = samples_out.at[i].set(sample_out)
        
        # add context cue-ing
        context = random_order[i]
        samples_in = samples_in.at[i, 3+context, :].set(jnp.ones((T,)))

    if not fix_output: samples_out = samples_out[:, 1:, :]

    return random_order, samples_in, samples_out

def one_of_each(key, T, fix_output=False):
    task_list = [sample_delay_pro, sample_delay_anti, sample_memory_pro, sample_memory_anti]
    order = jnp.arange(4)
    num_trials = 4
    num_tasks = num_trials
   
    sample_keys_trials = jr.split(key, num_trials)

    samples_in = jnp.zeros((num_trials, 3 + num_tasks, T))
    samples_out = jnp.zeros((num_trials, 3, T))
    for i in range(num_trials):
        sample_in, sample_out = task_list[order[i]](sample_keys_trials[i], T)
        samples_in = samples_in.at[i, :3, :].set(sample_in)
        samples_out = samples_out.at[i].set(sample_out)
        
        # add context cue-ing
        context = order[i]
        samples_in = samples_in.at[i, 3+context, :].set(jnp.ones((T,)))

    if not fix_output: samples_out = samples_out[:, 1:, :]

    return task_list, samples_in, samples_out

def sample_dm1(key, T):
    # generate thetas
    key_stim1, key_stim2, key_trialsplit, key_coherence  = jr.split(key, 4)
    theta_in1 = jr.uniform(key_stim1, minval=0., maxval=2*jnp.pi)
    theta_in2 = theta_in1 + jr.uniform(key_stim2, minval=jnp.pi/2, maxval=(3/2)*jnp.pi) # theta_2 sampled between 90 and 270 deg away from theta_1

    # segment trial into different periods
    key_stim1_on, key_stim1_off, key_stim2_off, key_response = jr.split(key_trialsplit,4)
    t_stim1_on = jax.random.uniform(key_stim1_on, minval=0.3, maxval=0.7) # time between 0 and when the stimulus1 comes on
    t_stim1_off = t_stim1_on + jax.random.uniform(key_stim1_off, minval=0.2, maxval=1.6) # time between when the stimulus1 comes on and goes off
    t_stim2_off = t_stim1_off + jax.random.uniform(key_stim2_off, minval=0.2, maxval=1.6) # time between when the stimulus2 comes on and goes off
    t_fix_off = t_stim2_off #+ jax.random.uniform(key_fix_off, minval=0.2, maxval=1.6) # time between when the stimulus comes on and when the go cue arrives (fix-off)
    t_response = t_fix_off + jax.random.uniform(key_response, minval=0.3, maxval=0.7) # time between go cue and end of trial

    t_total = t_response # total sampled time of trial

    t_stim1_on = t_stim1_on/t_total * T
    t_stim1_off = t_stim1_off/t_total * T
    t_fix_off = t_fix_off/t_total * T

    t = jnp.arange(T)
    # make fixation input
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [1., 0.]
    fix_input = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make stim inputs
    coherence = jax.random.uniform(key_coherence, minval=0.1, maxval=0.8) * jax.random.choice(key_coherence, a=jnp.array([-1,1]))
    stim_scale_1 = 1. + coherence
    stim_scale_2 = 1. - coherence
    cond_list = [t<t_stim1_on, jnp.logical_and(t>=t_stim1_on, t<t_stim1_off), jnp.logical_and(t>=t_stim1_off, t<t_fix_off), t >= t_fix_off]
    func_list_1 = [0, stim_scale_1 * jnp.sin(theta_in1), stim_scale_2 * jnp.sin(theta_in2), 0]
    func_list_2 = [0, stim_scale_1 * jnp.cos(theta_in1), stim_scale_2 * jnp.cos(theta_in2), 0]
    stim_input_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1] + cond_list[2]*func_list_1[2] + cond_list[3]*func_list_1[3]
    stim_input_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1] + cond_list[2]*func_list_2[2] + cond_list[3]*func_list_2[3]
    stim_input = jnp.array([stim_input_1, stim_input_2])

    # make fixation output
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [0.8, 0]
    fix_output = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make response outputs
    cond_list = [t < t_fix_off, t >=t_fix_off]
    if coherence > 0: # first stim is stronger
        func_list_1 = [0, jnp.sin(theta_in1)]
        func_list_2 = [0, jnp.cos(theta_in1)]
    if coherence < 0: # second stim is stronger
        func_list_1 = [0, jnp.sin(theta_in2)]
        func_list_2 = [0, jnp.cos(theta_in2)]
    response_output_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1]
    response_output_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1]
    response_output = jnp.array([response_output_1, response_output_2])

    ignore_input = jnp.zeros_like(stim_input) # second modality to be ignored

    return jnp.vstack((fix_input, stim_input, ignore_input)), jnp.vstack((fix_output, response_output))

def sample_dm2(key, T):
    # generate thetas
    key_stim1, key_stim2, key_trialsplit, key_coherence  = jr.split(key, 4)
    theta_in1 = jr.uniform(key_stim1, minval=0., maxval=2*jnp.pi)
    theta_in2 = theta_in1 + jr.uniform(key_stim2, minval=jnp.pi/2, maxval=(3/2)*jnp.pi) # theta_2 sampled between 90 and 270 deg away from theta_1

    # segment trial into different periods
    key_stim1_on, key_stim1_off, key_stim2_off, key_response = jr.split(key_trialsplit,4)
    t_stim1_on = jax.random.uniform(key_stim1_on, minval=0.3, maxval=0.7) # time between 0 and when the stimulus1 comes on
    t_stim1_off = t_stim1_on + jax.random.uniform(key_stim1_off, minval=0.2, maxval=1.6) # time between when the stimulus1 comes on and goes off
    t_stim2_off = t_stim1_off + jax.random.uniform(key_stim2_off, minval=0.2, maxval=1.6) # time between when the stimulus2 comes on and goes off
    t_fix_off = t_stim2_off #+ jax.random.uniform(key_fix_off, minval=0.2, maxval=1.6) # time between when the stimulus comes on and when the go cue arrives (fix-off)
    t_response = t_fix_off + jax.random.uniform(key_response, minval=0.3, maxval=0.7) # time between go cue and end of trial

    t_total = t_response # total sampled time of trial

    t_stim1_on = t_stim1_on/t_total * T
    t_stim1_off = t_stim1_off/t_total * T
    t_fix_off = t_fix_off/t_total * T

    t = jnp.arange(T)
    # make fixation input
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [1., 0.]
    fix_input = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make stim inputs
    coherence = jax.random.uniform(key_coherence, minval=0.1, maxval=0.8) * jax.random.choice(key_coherence, a=jnp.array([-1,1]))
    stim_scale_1 = 1. + coherence
    stim_scale_2 = 1. - coherence
    cond_list = [t<t_stim1_on, jnp.logical_and(t>=t_stim1_on, t<t_stim1_off), jnp.logical_and(t>=t_stim1_off, t<t_fix_off), t >= t_fix_off]
    func_list_1 = [0, stim_scale_1 * jnp.sin(theta_in1), stim_scale_2 * jnp.sin(theta_in2), 0]
    func_list_2 = [0, stim_scale_1 * jnp.cos(theta_in1), stim_scale_2 * jnp.cos(theta_in2), 0]
    stim_input_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1] + cond_list[2]*func_list_1[2] + cond_list[3]*func_list_1[3]
    stim_input_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1] + cond_list[2]*func_list_2[2] + cond_list[3]*func_list_2[3]
    stim_input = jnp.array([stim_input_1, stim_input_2])

    # make fixation output
    cond_list = [t < t_fix_off, t >=t_fix_off]
    func_list = [0.8, 0]
    fix_output = cond_list[0]*func_list[0] + cond_list[1]*func_list[1]

    # make response outputs
    cond_list = [t < t_fix_off, t >=t_fix_off]
    if coherence > 0: # first stim is stronger
        func_list_1 = [0, jnp.sin(theta_in1)]
        func_list_2 = [0, jnp.cos(theta_in1)]
    if coherence < 0: # second stim is stronger
        func_list_1 = [0, jnp.sin(theta_in2)]
        func_list_2 = [0, jnp.cos(theta_in2)]
    response_output_1 = cond_list[0]*func_list_1[0] + cond_list[1]*func_list_1[1]
    response_output_2 = cond_list[0]*func_list_2[0] + cond_list[1]*func_list_2[1]
    response_output = jnp.array([response_output_1, response_output_2])

    ignore_input = jnp.zeros_like(stim_input) # first modality to be ignored

    return jnp.vstack((fix_input, ignore_input, stim_input)), jnp.vstack((fix_output, response_output))