import numpy as np
import matplotlib.pyplot as plt
import yaml

class parse(object):
    def __init__(self, path):
        with open(path, 'r') as file:
            self.parameters = yaml.safe_load(file)

    def __getitem__(self, key):
        return self.parameters[key]

    def save(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.parameters, f)

class recurrent_network:
    def __init__(self, init_params):
        self.dtype = np.float32
        self.time_steps = init_params['time_steps']
        self.neuron_num = init_params['neuron_num']
        self.threshold = init_params['threshold']
        self.lr = init_params['learning_rate']
        self.window_size = init_params['window_size']
        if init_params['func_type'] == 'exp':
            self.delta_mem = init_params['delta_mem']
            self.rho_0 = 1
        self.weight_matrix = np.random.normal(size=(self.neuron_num, self.neuron_num)).astype(self.dtype)*0.2
        row, col = np.diag_indices_from(self.weight_matrix)
        self.weight_matrix[row,col] = -self.threshold
        self.input_weight = np.random.normal(size=(self.neuron_num)).astype(self.dtype)*0.1
        self.membrane_potentials = np.zeros((self.neuron_num, self.time_steps), dtype=self.dtype)
        self.firing_rate = np.zeros((self.neuron_num, self.time_steps), dtype=self.dtype)
        self.spike_train = np.zeros((self.neuron_num, self.time_steps), dtype=np.int8)
        if init_params['input_type'] == 'normal_random':
            self.input = np.random.normal(size=(self.neuron_num, self.time_steps)).astype(self.dtype)*0.1
        elif init_params['input_type'] == 'const':
            self.input_level = init_params['input_level']
            self.input = self.input_level * np.ones((self.neuron_num, self.time_steps), dtype=self.dtype)
        self.psc = np.zeros((self.neuron_num, self.time_steps), dtype=self.dtype)
        self.tau_mem = init_params['tau_mem']
        self.mem_decay = 1 - 1/self.tau_mem
        self.tau_psc = init_params['tau_psc']
        self.psc_decay = 1 - 1/self.tau_psc
        self.fsize = 13
    def plot(self):
        
        size_factor = 0.12
        total_plots = 5
        name_list = ['membrane_potentials','firing_rate','spike_train','input','psc']
        value_list = [self.membrane_potentials,np.clip(self.firing_rate,0,1),self.spike_train,self.input,self.psc]
#         plt.figure(figsize=(30,3))
#         plt.imshow(self.weight_matrix)
#         plt.title('weight_matrix', fontsize = fsize)
        for i in [0,1,2]:
            plt.figure(figsize=(self.time_steps*size_factor/3, self.neuron_num*size_factor/3))
            plt.imshow(value_list[i])
            plt.title(name_list[i], fontsize = self.fsize)
    def forward(self):
        temp_mem = np.zeros(a.neuron_num)
        temp_psc = np.zeros(a.neuron_num)
        for t in range(self.time_steps):
            temp_mem = temp_mem*self.mem_decay + np.matmul(self.input[:,t].T,self.input_weight) + np.matmul(temp_psc.T, self.weight_matrix)
            self.membrane_potentials[:,t] = temp_mem.copy()
            self.firing_rate[:,t] = self.rho_0 * np.exp(np.clip((self.membrane_potentials[:,t]-self.threshold)/self.delta_mem, -100,1))
            prob = np.random.uniform(size=self.neuron_num)
            self.spike_train[:,t] = (self.firing_rate[:,t] > prob)
            self.psc[:,t] = temp_psc * self.psc_decay + 1/self.tau_psc * self.spike_train[:,t]
            temp_psc = self.psc[:,t]
    def STDP(self):
        self.weight_update = np.ones(shape=(self.neuron_num, self.neuron_num),dtype=np.float32)
        learning_rate = self.lr
        window_length = self.window_size
        window_t = np.arange(-window_length,window_length+1,1)
        STDP_func = 0.42*np.exp(-window_t**2/(10))*(np.sign(window_t)+0.5)
        STDP_repete = np.empty((self.neuron_num,2*window_length+1),dtype=np.float32)
        STDP_repete[:]=STDP_func
        pad_spikes = np.pad(self.spike_train,((0,0),(window_length,window_length)),'constant')
        for i in range(self.time_steps):
            updates = 1 + self.lr * pad_spikes[:,i:i+11] * STDP_repete
            start_index = np.where(pad_spikes[:,i+5]==1)
            updates = np.prod(updates,axis=1)
            end_index = np.where(updates!=1)
            self.weight_update[np.ix_(start_index[0],end_index[0])]*=updates[end_index]
        row, col = np.diag_indices_from(self.weight_update)
        self.weight_update[row,col] = 1.
        self.weight_update = np.clip(self.weight_update, 0, 2)
        self.weight_matrix = self.weight_matrix * self.weight_update 
        energy = np.linalg.norm(self.weight_matrix)
        factor = 8/energy
        self.weight_matrix *= factor
        row, col = np.diag_indices_from(self.weight_matrix)
        self.weight_matrix[row,col] = -self.threshold
    def STDP_sum(self):
        self.weight_update = np.ones(shape=(self.neuron_num, self.neuron_num),dtype=np.float32)
        learning_rate = self.lr
        window_length = self.window_size
        window_t = np.arange(-window_length,window_length+1,1)
        STDP_func = 0.42*np.exp(-window_t**2/(10))*(np.sign(window_t)+0.5)
        relative_effect = np.array([np.convolve(a.spike_train[i], STDP_func) for i in range(a.neuron_num)])
        updates = learning_rate * np.matmul(relative_effect[:,window_length:-window_length],self.spike_train.T)/self.neuron_num
        self.weight_matrix += updates
        self.weight_update = updates
        row, col = np.diag_indices_from(self.weight_update)
        self.weight_update[row,col] = 0.
        energy = np.linalg.norm(self.weight_matrix)
        factor = 8/energy
        self.weight_matrix *= factor
        row, col = np.diag_indices_from(self.weight_matrix)
        self.weight_matrix[row,col] = -self.threshold
    def STDP_plot(self):
        plt.figure(figsize=(12,3))
        plt.subplot(1,3,1)
        plt.imshow(self.weight_matrix)
        plt.title('new weight matrix', fontsize = self.fsize)
        plt.colorbar()
        plt.subplot(1,3,2)
        plt.imshow(self.weight_update)
        plt.title('weight updating signal', fontsize = self.fsize)
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.hist(self.weight_matrix[~np.eye(self.weight_matrix.shape[0],dtype=bool)].ravel(),bins=50)
        plt.title('weight distribution', fontsize = self.fsize)
    def auto_plot(self):
        plt.clf()
        plt.subplot(3,4,1)
        plt.imshow(self.weight_matrix)
        plt.title('new weight matrix', fontsize = self.fsize)
        plt.colorbar()
        plt.subplot(3,4,2)
        plt.imshow(self.weight_update)
        plt.title('weight updating signal', fontsize = self.fsize)
        plt.colorbar()
        plt.subplot(3,4,3)
        plt.hist(self.weight_matrix[~np.eye(self.weight_matrix.shape[0],dtype=bool)].ravel(),bins=50)
        plt.title('weight distribution', fontsize = self.fsize)
        plt.subplot(3,4,4)
        self.graph_plot()
        size_factor = 0.12
        total_plots = 5
        name_list = ['membrane_potentials','postsynaptic_current','spike_train','firing_rate','input']
        value_list = [self.membrane_potentials,self.psc,self.spike_train,np.clip(self.firing_rate,0,1),self.input]
        for i in [0,1]:
            plt.subplot(3,1,2+i)
            plt.imshow(value_list[i])
            plt.title(name_list[i], fontsize = self.fsize)
        plt.pause(0.05)
    def graph_plot(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(self.neuron_num))
        for i in range(self.neuron_num):
            for j in range(self.neuron_num):
                c = 'gold' if self.weight_matrix[i,j]>0 else 'navy'
                G.add_edge(i, j, color = c, weight = self.weight_matrix[i,j])
        weights = [G[u][v]['weight'] for u,v in G.edges]
        colors = [G[u][v]['color'] for u,v in G.edges]
        options = {
            'node_color': 'mediumseagreen',
            'width': np.abs(weights),
            'edge_color': colors,
            'node_size': 40,
        }
        nx.draw_circular(G, **options)
    def fast_auto_plot(self):
        plt.clf()
        plt.subplot(1,4,1)
        plt.imshow(self.weight_matrix)
        plt.title('new weight matrix', fontsize = self.fsize)
        plt.colorbar()
        plt.subplot(1,4,2)
        plt.imshow(self.weight_update)
        plt.title('weight updating signal', fontsize = self.fsize)
        plt.colorbar()
        plt.subplot(1,4,3)
        plt.hist(self.weight_matrix[~np.eye(self.weight_matrix.shape[0],dtype=bool)].ravel(),bins=50)
        plt.title('weight distribution', fontsize = self.fsize)
        plt.subplot(1,4,4)
        self.graph_plot()
        plt.pause(0.05)
        
plt.figure(figsize=(12,10))
# %matplotlib inline
param = parse('./config/test.yaml')

a = recurrent_network(param)
for i in range(500):
    a.forward()
#     a.plot()
    a.STDP()
    a.auto_plot()
#     a.STDP_plot()

a = recurrent_network(param)
a.forward()
a.plot()
a.STDP()
a.STDP_plot()