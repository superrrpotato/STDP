import torch
import global_v as glv

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
