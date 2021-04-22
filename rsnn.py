import global_v as glv
import torch
import torch.nn as nn
import functions.util_functions as f
import networkx as nx
class RSNN():
    def __init__(self, init_params):
        # super(RSNN, self).__init__()
        self.dtype = glv.dtype
        self.time_steps = init_params['time_steps']
        self.observed_neuron_num = init_params['observed_neuron_num']
        self.latent_neuron_num = init_params['latent_neuron_num']
        self.neuron_num = self.observed_neuron_num + self.latent_neuron_num
        self.threshold = init_params['threshold']
        self.window_size = init_params['window_size']
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
                self.neuron_num), dtype=glv.dtype, device=glv.device)*0.2
        row = col = range(self.neuron_num)
        self.total_weight_matrix[row,col] = -self.threshold
        self.input_weight = 0.1 + torch.rand(self.observed_neuron_num,\
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






