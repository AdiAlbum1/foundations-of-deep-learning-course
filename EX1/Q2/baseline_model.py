import torch.nn as nn
import torch.nn.functional as F

class Baseline_Network(nn.Module):
    def __init__(self, d_in=3072, d_out=10, net_width=256, n_hidden_layers=1, is_dropout=False, p_dropout=0):
        super(Baseline_Network, self).__init__()
        self.input_layer = nn.Linear(d_in, net_width)
        self.output_layer = nn.Linear(net_width, d_out)
        self.inner_layers = [nn.Linear(net_width, net_width) for i in range(n_hidden_layers-1)]
        self.is_dropout = is_dropout
        if self.is_dropout:
            self.dropout = nn.Dropout(p_dropout)
        self.p_dropout = p_dropout

    def normal_random_init(self, std=1.0):
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=std)
        nn.init.zeros_(self.input_layer.bias)
        for i in range(len(self.inner_layers)):
            nn.init.normal_(self.inner_layers[i].weight, mean=0.0, std=std)
            nn.init.zeros_(self.inner_layers[i].bias)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=std)
        nn.init.zeros_(self.output_layer.bias)

    def xavier_init(self):
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        for i in range(len(self.inner_layers)):
            nn.init.xavier_uniform_(self.inner_layers[i].weight)
            nn.init.zeros_(self.inner_layers[i].bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        hidden_activation = F.relu(self.input_layer(x))
        for i in range(len(self.inner_layers)):
            hidden_activation = F.relu(self.inner_layers[i](hidden_activation))
        if self.is_dropout:                         # dropout on last layer, if is_dropout=True
            hidden_activation = self.dropout(hidden_activation)
        output = self.output_layer(hidden_activation)
        return output
