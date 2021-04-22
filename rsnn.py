import global_v as glv
import torch
import torch.nn as nn
import torch.nn.functional as F
import functions.util_functions as f
import networkx as nx
import matplotlib.pyplot as plt
class RSNN():
    def __init__(self, init_params):
        # super(RSNN, self).__init__()
        self.dtype = glv.dtype
        self.time_steps = init_params['time_steps']
        self.observed_neuron_num = init_params['observed_neuron_num']
        self.latent_neuron_num = init_params['latent_neuron_num']
        self.neuron_num = self.observed_neuron_num + self.latent_neuron_num
        self.threshold = init_params['threshold']
        self.tau_mem = init_params['tau_mem']
        self.mem_decay = 1 - 1/self.tau_mem
        self.tau_psc = init_params['tau_psc']
        self.psc_decay = 1 - 1/self.tau_psc
        self.lr = init_params['learning_rate']
        self.window_size = init_params['window_size']
        if init_params['func_type'] == 'exp':
            self.delta_mem = init_params['delta_mem']
            self.rho_0 = init_params['rho_0']
        self.total_weight_matrix = torch.randn(size=(self.neuron_num,\
                self.neuron_num), dtype=glv.dtype, device=glv.device)
        row = col = range(self.neuron_num)
        self.total_weight_matrix[row,col] = -self.threshold
        self.weight_matrix=self.total_weight_matrix[:self.observed_neuron_num]\
                [:self.observed_neuron_num]
        self.init_observed_energy = torch.norm(self.weight_matrix)
        self.input_weight = 0.5 + torch.rand(self.observed_neuron_num,\
                dtype=glv.dtype, device=glv.device) * 0.1
        self.latent_bias = torch.rand(self.latent_neuron_num,\
                dtype=glv.dtype, device=glv.device) * 0.1
        self.membrane_potentials = torch.zeros((self.neuron_num,\
                self.time_steps), dtype=glv.dtype, device=glv.device)
        self.firing_rate = torch.zeros((self.neuron_num,\
                self.time_steps), dtype=glv.dtype, device=glv.device)
        self.spike_train = torch.zeros((self.neuron_num,\
                self.time_steps), dtype=torch.bool, device=glv.device)
        self.psc = torch.zeros((self.neuron_num, self.time_steps),\
                dtype=glv.dtype, device=glv.device)
    def forward(self, spike_input):
        temp_mem = torch.zeros(self.neuron_num, device=glv.device)
        temp_psc = torch.zeros(self.neuron_num, device=glv.device)
        spikes = f.psp(spike_input, self.tau_psc)
        for t in range(self.time_steps):
            temp_mem = temp_mem*self.mem_decay\
                    + torch.matmul(temp_psc.T, self.total_weight_matrix)
            # Not sure whethere we need input weights or not.
            temp_mem[:self.observed_neuron_num] += \
                    torch.matmul(spikes[:, t].T, self.input_weight)
            temp_mem[self.observed_neuron_num:] += self.latent_bias
            self.membrane_potentials[:, t] = temp_mem.clone()
            self.firing_rate[:, t] = self.rho_0 *\
                    torch.exp(torch.clamp((self.membrane_potentials[:,t]\
                    -self.threshold)/self.delta_mem, -100, 1))
            prob = torch.rand(self.neuron_num, device=glv.device)
            self.spike_train[:, t] = (self.firing_rate[:,t] > prob)
            self.psc[:, t] = temp_psc * self.psc_decay + 1/self.tau_psc * self.spike_train[:,t]
            temp_psc = self.psc[:, t]
            temp_mem = temp_mem * (1 - self.spike_train[:, t].int())
    def stdp_update(self):
        self.weight_update = torch.ones((self.observed_neuron_num,\
            self.observed_neuron_num), dtype=glv.dtype, device=glv.device)
        window_length = self.window_size
        window_t = torch.arange(-window_length,window_length+1,1,\
                dtype=glv.dtype)
        STDP_func = 0.42*torch.exp(-window_t**2/(10))*(torch.sign(window_t)+0.5)
        STDP_repete = torch.empty((self.neuron_num,2*window_length+1),\
                dtype=glv.dtype)
        STDP_repete[:]=STDP_func
        pad_spikes = F.pad(self.spike_train,(window_length, window_length, 0, 0))
        for i in range(self.time_steps):
            updates = 1 + self.lr * pad_spikes[:,i:i+11] * STDP_repete
            start_index = torch.where(pad_spikes[:,i+5]==1) # for all spike exist
            updates = torch.prod(updates,axis=1)
            end_index = torch.where(updates!=1) # for all update exist
            self.weight_update[torch.meshgrid(start_index[0],end_index[0])]\
                    *=updates[end_index] # accumulate the update on
        row = col = range(self.observed_neuron_num)
        self.weight_update[row,col] = 1.
        self.weight_update = torch.where(self.weight_matrix<0,\
                1/self.weight_update, self.weight_update) # inhibitory link
        self.weight_update = torch.clamp(self.weight_update, 0, 2)
        self.weight_matrix = self.weight_matrix * self.weight_update
        energy = torch.norm(self.weight_matrix)
        factor = self.init_observed_energy/energy
        self.weight_matrix *= factor # energy normalization
        row = col = range(self.observed_neuron_num)
        self.weight_matrix[row,col] = -self.threshold
        self.total_weight_matrix[:self.observed_neuron_num, \
                :self.observed_neuron_num] = self.weight_matrix[row,col]
    def cellular_visualize(self):
        self.connection_map = torch.zeros((glv.length, glv.length))
        for i in range(glv.non_zero_num):
            start_neuron_colu = glv.line_index[i] % glv.length
            start_neuron_line = glv.line_index[i] // glv.length
            target_neuron_colu = glv.colum_index[i] % glv.length
            target_neuron_line = glv.colum_index[i] // glv.length
            self.connection_map[start_neuron_colu, start_neuron_line]\
                    += self.weight_matrix[glv.line_index[i], glv.colum_index[i]]
            self.connection_map[target_neuron_colu, target_neuron_line]\
                    += self.weight_matrix[glv.line_index[i], glv.colum_index[i]]
        plt.imshow(self.connection_map)
        plt.colorbar()
        #plt.show()
    """
    def graph_plot(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(self.observed_neuron_num))
        for i in range(self.observed_neuron_num):
            for j in range(self.observed_neuron_num):
                c = 'gold' if self.weight_matrix[i,j]>0 else 'navy'
                G.add_edge(i, j, color = c, weight = self.weight_matrix[i,j])
        weights = [G[u][v]['weight'] for u,v in G.edges]
        colors = [G[u][v]['color'] for u,v in G.edges]
        pos = None
        options = {
            'node_color': 'mediumseagreen',
            'width': np.abs(weights),
            'edge_color': colors,
            'node_size': 40,
            'pos': pos
        }
        nx.draw(G, **options)
    """






