import torch


def calc_end_to_end_matrix(net):
    end_to_end_mat = torch.tensor([1.0])
    network_parameters = net.parameters()
    for net_param in network_parameters:
        end_to_end_mat = torch.matmul(net_param.float(), end_to_end_mat)
    end_to_end_mat = float(end_to_end_mat)      # output is 1d
    return end_to_end_mat
