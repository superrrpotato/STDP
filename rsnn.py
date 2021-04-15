import global_v as glv
import torch
class recurrent_network:
    def __init__(self, init_params):
        self.dtype = glv.dtype
        self.time_steps = init_params['time_steps']
        self.observed_neuron_num = init_params['neuron_num']
        self.latent_neuron_num = init_params['latent_neuron_num']
        self.threshold = init_params['threshold']
        self.lr = init_params['learning_rate']
        self.window_size = init_params['window_size']
        if init_params['func_type'] == 'exp':
            self.delta_mem = init_params['delta_mem']
            self.rho_0 = init_params['rho_0']


