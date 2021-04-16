import torch
import global_v as glv
def psp(inputs, tau_psc):
    shape = inputs.shape
    # n_steps = network_config['n_steps']
    # tau_s = network_config['tau_s']

    syn = torch.zeros(shape[0], dtype=glv.dtype, device=glv.device)
    syns = torch.zeros(shape, dtype=glv.dtype, device=glv.device)

    for t in range(shape[1]):
        syn = syn * ( 1 - 1 / tau_psc) + (1 / tau_psc) * inputs[:, t]
        syns[:, t] = syn

    return syns
