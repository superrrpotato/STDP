import global_v as glv
import torch
import torch.nn as nn
import functions.stdp as stdp
class RSNN(nn.Module):
    def __init__(self, init_params):
        super(RSNN, self).__init__()
        self.dtype = glv.dtype
        self.time_steps = init_params['time_steps']
        self.observed_neuron_num = init_params['neuron_num']
        self.latent_neuron_num = init_params['latent_neuron_num']
        self.neuron_num = self.observed_neuron_num + self.latent_neuron_num
        self.threshold = init_params['threshold']
        #self.lr = init_params['learning_rate']
        #self.window_size = init_params['window_size']
        #if init_params['func_type'] == 'exp':
        #    self.delta_mem = init_params['delta_mem']
        #    self.rho_0 = init_params['rho_0']
        self.total_weight_matrix = torch.randn(size=(self.neuron_num,\
            self.neuron_num), dtype=glv.dtype)*0.2
        row, col = range(self.neuron_num)
        self.total_weight_matrix[row,col] = -self.threshold
        self.input_weight = torch.randn(size=self.observed_neuron_num,\
                dtype=glv.dtype) * 0.1
    def forward(self, spike_input):
        observed_spike = stdp.STDP.apply(spike_input, init_params)
        return observed_spike





