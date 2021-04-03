import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Network(nn.Module):
    def __init__(self, p_dropout=-1, net_width=(64, 16)):
        super(Conv_Network, self).__init__()
        self.net_width = net_width
        self.conv_layer_1 = nn.Conv2d(3, self.net_width[0], 3)
        self.max_pooling_2 = nn.MaxPool2d(2, stride=2)
        self.conv_layer_3 = nn.Conv2d(self.net_width[0], self.net_width[1], 3)
        self.max_pooling_4 = nn.MaxPool2d(2, stride=2)
        self.fc_5 = nn.Linear(self.net_width[1] * 6 * 6, 784)
        if p_dropout >= 0:
          self.dropout = nn.Dropout(p_dropout)
        else:
          self.dropout = None
        self.fc_6 = nn.Linear(784, 10)

    def random_init(self, std=1.0):
        self.conv_layer_1.weight.data = self.conv_layer_1.weight.data.normal_(mean=0.0, std=std).to(torch.device("cuda:0"))
        self.conv_layer_1.bias.data = nn.init.zeros_(self.conv_layer_1.bias.data).to(torch.device("cuda:0"))
        self.conv_layer_3.weight.data = self.conv_layer_3.weight.data.normal_(mean=0.0, std=std).to(torch.device("cuda:0"))
        self.conv_layer_3.bias.data = nn.init.zeros_(self.conv_layer_3.bias.data).to(torch.device("cuda:0"))
        self.fc_5.weight.data = nn.init.normal_(self.fc_5.weight, mean=0.0, std=std).to(torch.device("cuda:0"))
        self.fc_5.bias.data = nn.init.zeros_(self.fc_5.bias).to(torch.device("cuda:0"))
        self.fc_6.weight.data = nn.init.normal_(self.fc_6.weight, mean=0.0, std=std).to(torch.device("cuda:0"))
        self.fc_6.bias.data = nn.init.zeros_(self.fc_6.bias).to(torch.device("cuda:0"))
      
    def xavier_init(self):
        nn.init.xavier_uniform_(self.conv_layer_1.weight.data)
        nn.init.zeros_(self.conv_layer_1.bias.data)

        nn.init.xavier_uniform_(self.conv_layer_3.weight.data)
        nn.init.zeros_(self.conv_layer_3.bias.data)

        nn.init.xavier_uniform_(self.fc_5.weight.data)
        nn.init.zeros_(self.fc_5.bias.data)

        nn.init.xavier_uniform_(self.fc_6.weight.data)
        nn.init.zeros_(self.fc_6.bias.data)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).to(torch.device("cuda:0"))
        hidden_activation_1 = F.relu(self.conv_layer_1(x))
        hidden_activation_2 = self.max_pooling_2(hidden_activation_1)
        hidden_activation_3 = F.relu(self.conv_layer_3(hidden_activation_2))
        hidden_activation_4 = self.max_pooling_4(hidden_activation_3)
        hidden_activation_5 = self.fc_5(hidden_activation_4.reshape(-1, self.net_width[1] * 6 * 6))
        if self.dropout:
          hidden_activation_5 = self.dropout(hidden_activation_5)
        output = self.fc_6(hidden_activation_5)
        return output