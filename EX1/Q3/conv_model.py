import torch.nn as nn
import torch.nn.functional as F

class Conv_Network(nn.Module):
    def __init__(self):
        super(Conv_Network, self).__init__()
        self.conv_layer_1 = nn.Conv2d(3, 64, 3)
        self.max_pooling_2 = nn.MaxPool2d(2, stride=2)
        self.conv_layer_3 = nn.Conv2d(64, 16, 3)
        self.max_pooling_4 = nn.MaxPool2d(2, stride=2)
        self.fc_5 = nn.Linear(16 * 6 * 6, 784)
        self.fc_6 = nn.Linear(784, 10)

    def random_init(self, std=1.0):
        self.conv_layer_1.weight.data.normal_(mean=0.0, std=std)
        nn.init.zeros_(self.conv_layer_1.bias.data)
        self.conv_layer_3.weight.data.normal_(mean=0.0, std=std)
        nn.init.zeros_(self.conv_layer_3.bias.data)
        nn.init.normal_(self.fc_5.weight, mean=0.0, std=std)
        nn.init.zeros_(self.fc_5.bias)
        nn.init.normal_(self.fc_6.weight, mean=0.0, std=std)
        nn.init.zeros_(self.fc_6.bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        hidden_activation_1 = F.relu(self.conv_layer_1(x))
        hidden_activation_2 = self.max_pooling_2(hidden_activation_1)
        hidden_activation_3 = F.relu(self.conv_layer_3(hidden_activation_2))
        hidden_activation_4 = self.max_pooling_4(hidden_activation_3)
        hidden_activation_5 = self.fc_5(hidden_activation_4.view(-1, 16 * 6 * 6))
        output = self.fc_6(hidden_activation_5)
        return output