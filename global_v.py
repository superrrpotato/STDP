import torch

def init(dev, dty, params):
    global dtype, device, time_steps, threshold, observed_neuron_num, \
            latent_neuron_num, tau_mem, tau_psc, delta_mem, lr, window_size,\
    dtype = dty
    device = dev
    time_steps = params['time_steps']
    threshold = params['threshold']
    observed_neuron_num = params['observed_neuron_num']
    latent_neuron_num = params['latent_neuron_num']
    tau_mem = params['tau_mem']
    tau_psc = params['tau_psc']
    delta_mem = params['delta_mem']
    lr = params['learning_rate']
    window_size = params['window_size']
