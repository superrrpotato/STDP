import os
import torch
import time
from network_parser import parse
import logging
import argparse
import global_v as glv
from rsnn import RSNN
from utils import aboutCudaDevices
#import pycuda.driver as cuda
import matplotlib.pyplot as plt
from datasets import loadMNIST
import cell
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config',\
            help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint',\
            help='The path of checkpoint, if use checkpoint')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config

    logging.basicConfig(filename='result.log', level=logging.INFO)

    logging.info("start parsing settings")

    params = parse(config_path)

    logging.info("finish parsing settings")

    if torch.cuda.is_available():
        device = params['device']#torch.device("cuda")
        cuda.init()
        c_device = aboutCudaDevices()
        print(c_device.info())
        print("selected device: ", device)
    else:
        device = torch.device("cpu")
        print("No GPU is found")
    dtype = torch.float32
    glv.init(device, dtype, params)
    batch_size = params['batch_size']
    if params['dataset'] == "MNIST":
        data_path = os.path.expanduser(params['data_path'])
        train_loader, test_loader = loadMNIST.get_mnist(data_path,\
                batch_size)
    new_rsnn = RSNN(params)
    #counter=0
    plt.figure(figsize=(12,2.5))
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        #if counter == 1:
        #    break
        inputs = inputs.view(-1)
        inputs = inputs.to(glv.device)
        inputs.type(glv.dtype)
        input_spikes = inputs.unsqueeze_(-1).repeat(1,params['time_steps'])
        new_rsnn.forward(input_spikes*50)
        new_rsnn.stdp_update()
        plt.clf()
        new_rsnn.neuron_spike_visualize(batch_idx)
        plt.pause(0.5)
        #counter += 1
    #cell.cellular_weight_visualize(inputs, params)
    #cell.spike_visualize(inputs, params)
    """
    plt.figure()
    plt.imshow(inputs.view(28,28))
    inputs = inputs.view(-1)
    inputs = inputs.to(device)
    inputs.type(dtype)
    input_spikes = inputs.unsqueeze_(-1).repeat(1,params['time_steps'])
    start_time = time.time()
    plt.figure()
    for i in range(100):
        print(i)
        new_rsnn.forward(input_spikes*5)
        new_rsnn.stdp_update()
        plt.clf()
        new_rsnn.cellular_visualize()
        plt.pause(0.05)
    """
    #print("--- %s seconds ---" % (time.time() - start_time))
    # plt.imshow(new_rsnn.spike_train.cpu())
    #plt.show()
