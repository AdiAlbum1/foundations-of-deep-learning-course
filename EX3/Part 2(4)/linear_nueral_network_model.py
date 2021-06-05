import torch.nn as nn
import torch.nn.functional as F
import torch

class Linear_Network(nn.Module):
    def __init__(self, D_in, D_hidden, D_out, N=2):
        super(Linear_Network, self).__init__()
        assert D_hidden <= min(D_in, D_out)
        self.N = N
        self.input_layer = nn.Linear(D_in, D_hidden, bias=True)
        if N>=3:
            self.first_hidden_layer = nn.Linear(D_hidden, D_hidden, bias=True)
        if N>=4:
            self.second_hidden_layer = nn.Linear(D_hidden, D_hidden, bias=True)

        self.output_layer = nn.Linear(D_hidden, D_out, bias=True)
        
    def normal_random_init(self, std=1.0):
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=std)

        if self.N >= 3:
            nn.init.normal_(self.first_hidden_layer.weight, mean=0.0, std=std)
        if self.N >= 4:
            nn.init.normal_(self.second_hidden_layer.weight, mean=0.0, std=std)

        nn.init.normal_(self.output_layer.weight, mean=0.0, std=std)

    def xavier_init(self, std=1.0):
        nn.init.xavier_uniform_(self.input_layer.weight, mean=0.0, std=std)
        nn.init.xavier_uniform_(self.input_layer.bias, mean=0.0, std=std)

        if self.N >= 3:
            nn.init.xavier_uniform_(self.first_hidden_layer.weight, mean=0.0, std=std)
            nn.init.xavier_uniform_(self.first_hidden_layer.bias, mean=0.0, std=std)
        if self.N >= 4:
            nn.init.xavier_uniform_(self.second_hidden_layer.weight, mean=0.0, std=std)
            nn.init.xavier_uniform_(self.second_hidden_layer.bias, mean=0.0, std=std)

        nn.init.xavier_uniform_(self.output_layer.weight, mean=0.0, std=std)
        nn.init.xavier_uniform_(self.output_layer.bias, mean=0.0, std=std)

    def forward(self, x):
        layer = self.input_layer(x)

        if self.N >= 3:
            layer = self.first_hidden_layer(layer)
        if self.N >= 4:
            layer = self.second_hidden_layer(layer)

        output = self.output_layer(layer)
        return output
