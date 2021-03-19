import torch.nn as nn
import torch.nn.functional as F

class Baseline_Network(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Baseline_Network, self).__init__()
        self.input_layer = nn.Linear(D_in, H)
        self.output_layer = nn.Linear(H, D_out)

    def forward(self, x):
        hidden_activation = F.relu(self.input_layer(x))
        output = self.output_layer(hidden_activation)
        return output