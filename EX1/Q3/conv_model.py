import torch.nn as nn
import torch.nn.functional as F

class Conv_Network(nn.Module):
    def __init__(self):
        super(Conv_Network, self).__init__()
        self.conv_layer_1 = nn.Conv2d(3, 64, 3)
        self.max_pooling_2 = nn.MaxPool2d(2, stride=2)
        self.conv_layer_3 = nn.Conv2d(3, 16, 3)
        self.max_pooling_4 = nn.MaxPool2d(2, stride=2)
        self.fc_5 = nn.Linear(16 * 6 * 6, 784)          # TODO - fix input dimensions
        self.fc_6 = nn.Linear(784, 10)

    def random_init(self, std=1.0):
        nn.init.normal_(self.conv_layer_1.weight, mean=0.0, std=std)
        nn.init.normal_(self.conv_layer_1.bias, mean=0.0, std=std)
        nn.init.normal_(self.conv_layer_3.weight, mean=0.0, std=std)
        nn.init.normal_(self.conv_layer_3.bias, mean=0.0, std=std)

        nn.init.normal_(self.fc_5.weight, mean=0.0, std=std)
        nn.init.normal_(self.fc_5.bias, mean=0.0, std=std)
        nn.init.normal_(self.fc_6.weight, mean=0.0, std=std)
        nn.init.normal_(self.fc_6.bias, mean=0.0, std=std)

    def forward(self, x):
        hidden_activation_1 = F.relu(self.conv_layer_1(x))
        hidden_activation_2 = self.max_pooling_2(hidden_activation_1)
        hidden_activation_3 = F.relu(self.conv_layer_3(hidden_activation_2))
        hidden_activation_4 = self.max_pooling_4(hidden_activation_3)
        hidden_activation_5 = self.fc_5(hidden_activation_4.view(-1, self.num_flat_features(hidden_activation_4)))
        # TODO - NO ACTIVATION? Check forum for answer
        output = self.fc_6(hidden_activation_5)
        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features