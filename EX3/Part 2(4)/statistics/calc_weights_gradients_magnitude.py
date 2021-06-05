import torch
import numpy as np

def calc_weights_gradients_magnitude(net):
    grads = []
    for param in net.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    weights_gradients_magnitude = np.linalg.norm(np.array(grads))
    return weights_gradients_magnitude