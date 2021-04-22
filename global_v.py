import torch
import math
import cell
def init(dev, dty, params):
    global dtype, device, line_index, length, colum_index, non_zero_num
    dtype = dty
    device = dev
    length = int(math.sqrt(params['observed_neuron_num']))
    line_index, colum_index = cell.get_cellular_index(length)
    non_zero_num = len(line_index)
    # time_steps = params['time_steps']
    # threshold = params['threshold']
    # observed_neuron_num = params['observed_neuron_num']
    # latent_neuron_num = params['latent_neuron_num']
    # tau_mem = params['tau_mem']
    # tau_psc = params['tau_psc']
    # delta_mem = params['delta_mem']
    # lr = params['learning_rate']
    # window_size = params['window_size']
