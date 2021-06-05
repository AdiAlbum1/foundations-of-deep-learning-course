import torch
import numpy as np

def end_to_end_dynamics_weight_update(net):
    weights_vector = obtain_weights_vector(net)
    gradients_vector = obtain_gradients_vector(net)

    updated_weights_vector = calculate_updated_weights_vector(weights_vector, gradients_vector)
    return net

def calculate_updated_weights_vector(weights_vector, gradients_vector):
    weights_vector = torch.reshape(weights_vector, (len(weights_vector), 1))
    weights_vector_transposed = torch.transpose(weights_vector, 0, 1)

    w_times_w_transpose = torch.matmul(weights_vector, weights_vector_transposed)
    w_transposed_times_w = torch.matmul(weights_vector_transposed, weights_vector)

    # SVD decompose w_times_w_transpose
    u, s, vh = np.linalg.svd(w_times_w_transpose)

    

    print(w_times_w_transpose.shape)
    print(w_transposed_times_w.shape)

def obtain_weights_vector(net):
    weights_vector = torch.zeros(0)
    weights = net.parameters()
    for weight in weights:
        weights_vector = torch.cat((weights_vector, torch.flatten(weight.data)), 0)

    return weights_vector

def obtain_gradients_vector(net):
    grads = []
    for param in net.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)

    return grads