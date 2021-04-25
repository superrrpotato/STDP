import torch
import global_v as glv
import matplotlib.pyplot as plt
from rsnn import RSNN

def get_cellular_index(length):
    line_index = []
    colum_index = []
    for i in range(length):
        for j in range(length):
            if j!=length-1:
                line_index += [length*i+j]
                colum_index += [length*i+j+1]
            if j!= 0:
                line_index += [length*i+j]
                colum_index += [length*i+j-1]
            if i!= length-1:
                line_index += [length*i+j]
                colum_index += [length*(i+1)+j]
            if i!= 0:
                line_index += [length*i+j]
                colum_index += [length*(i-1)+j]
    return torch.tensor(line_index).to(glv.device),\
            torch.tensor(colum_index).to(glv.device)
def cellular_weight_visualize(inputs, params):
    plt.figure(figsize=(5,5))
    plt.imshow(inputs.view(glv.length, glv.length))
    plt.figure(figsize=(5,5))
    inputs = inputs.view(-1)
    inputs = inputs.to(glv.device)
    inputs.type(glv.dtype)
    input_spikes = inputs.unsqueeze_(-1).repeat(1,params['time_steps'])
    new_rsnn = RSNN(params)
    for i in range(100):
        print(i)
        new_rsnn.forward(input_spikes*50)
        new_rsnn.stdp_update()
        plt.clf()
        new_rsnn.cellular_visualize(i)
        plt.pause(0.5)
def spike_visualize(inputs, params):
    plt.figure(figsize=(15,2.5))
    inputs = inputs.view(-1)
    inputs = inputs.to(glv.device)
    inputs.type(glv.dtype)
    input_spikes = inputs.unsqueeze_(-1).repeat(1,params['time_steps'])
    new_rsnn = RSNN(params)
    for i in range(10):
        print(i)
        new_rsnn.forward(input_spikes*5)
        new_rsnn.neuron_spike_visualize(i)
        new_rsnn.stdp_update()

