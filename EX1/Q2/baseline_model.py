import torch.nn as nn
import torch.nn.functional as F

class Baseline_Network(nn.Module):
    def __init__(self, D_in, H, D_out, is_dropout=False, p_dropout=0):
        super(Baseline_Network, self).__init__()
        self.input_layer = nn.Linear(D_in, H)
        self.output_layer = nn.Linear(H, D_out)
        self.is_dropout = is_dropout
        if self.is_dropout:
            self.dropout = nn.Dropout(p_dropout)
        self.p_dropout = p_dropout

    def normal_random_init(self, std=1.0):
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=std)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=std)
        nn.init.zeros_(self.output_layer.bias)

    def xavier_init(self):
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        hidden_activation = F.relu(self.input_layer(x))
        if self.is_dropout:
            hidden_activation = self.dropout(hidden_activation)
        output = self.output_layer(hidden_activation)
        return output
