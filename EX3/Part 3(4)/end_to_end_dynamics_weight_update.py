import torch
import numpy as np


def end_to_end_dynamics_weight_update(net, N, eta):
    network_parameters = net.parameters()
    new_network_parameters = []

    # calculate new network parameters
    for network_param in network_parameters:
        w = network_param
        grad_of_l_at_w = network_param.grad  # obtain gradients

        w_transpose = torch.transpose(w, 0, 1)

        w_times_w_transpose = torch.matmul(w, w_transpose)
        w_transpose_times_w = torch.matmul(w_transpose, w)

        partial_sum = 0
        for j in range(1, N + 1):
            # SVD decompose w_times_w_transpose
            u, s, vh = np.linalg.svd(np.array(w_times_w_transpose.detach()))
            if j-1 == 0:
                w_times_w_transpose_powered = np.identity(len(s))
            else:
                # power each element on s's diagonal by (j-1)/N
                s = np.power(s, ((j - 1) / N))
                w_times_w_transpose_powered = np.matmul(np.matmul(u, np.diag(s)), vh)

            # SVD decompose w_transpose_times_w
            u, s, vh = np.linalg.svd(np.array(w_transpose_times_w.detach()))
            if N-j == 0:
                w_transpose_times_w_powered = np.identity(len(s))
            else:
                # power each element on s's diagonal by (N-j)/N
                s = np.power(s, (N - j) / N)
                w_transpose_times_w_powered = np.matmul(np.matmul(u, np.diag(s)), vh)

            curr_elem_in_sum = np.matmul(np.matmul(w_times_w_transpose_powered, grad_of_l_at_w),
                                         w_transpose_times_w_powered)

            partial_sum = np.add(partial_sum, curr_elem_in_sum)

        new_network_param = network_param - eta * partial_sum
        new_network_parameters.append(new_network_param)

    # perform network parameters update
    for i, p in enumerate(net.parameters()):
        p.data = new_network_parameters[i].float()

    return net
