import torch.nn as nn
import torch.nn.functional as F

class Baseline_Network(nn.Module):
    def __init__(self, D_in, D_hidden, D_out, N=2):
        super(Baseline_Network, self).__init__()
        assert D_hidden >= min(D_in, D_out)
        self.N = N
        self.input_layer = nn.Linear(D_in, D_hidden, bias=False)
        if N==3:
          self.first_hidden_layer = nn.Linear(D_hidden, D_hidden, bias=False)
        if N==4:
          self.second_hidden_layer = nn.Linear(D_hidden, D_hidden, bias=False)

        self.output_layer = nn.Linear(D_hidden, D_out, bias=False)
        
    def normal_random_init(self, std=1.0):
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=std)

        if self.N == 3:
          nn.init.normal_(self.first_hidden_layer.weight, mean=0.0, std=std)
        if self.N == 4:
          nn.init.normal_(self.second_hidden_layer.weight, mean=0.0, std=std)

        nn.init.normal_(self.output_layer.weight, mean=0.0, std=std)

    def xavier_init(self):
        nn.init.xavier_uniform_(self.input_layer.weight)

        if self.N == 3:
          nn.init.xavier_uniform_(self.first_hidden_layer.weight, mean=0.0, std=std)
        if self.N == 4:
          nn.init.xavier_uniform_(self.second_hidden_layer.weight, mean=0.0, std=std)

        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        first = self.input_layer(x)

        if self.N == 3:
          first = self.first_hidden_layer(first)
        if self.N == 3:
          first = self.second_hidden_layer(first)

        output = self.output_layer(hidden_activation)
        return output
