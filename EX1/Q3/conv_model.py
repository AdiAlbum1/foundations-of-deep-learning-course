import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Network(nn.Module):
    def __init__(self, p_dropout=-1, net_width=(64, 16), k=2):
        """
        p_droput: what is the probabilty to drop a neuron in the network
        net_width: how much filters each conv layer have
        k: number of conv layers, must hold 2<=k<=5
        """
        super(Conv_Network, self).__init__()
        self.net_width = net_width
        self.k = k

        if self.k>=3:
          self.conv_layer_0 = nn.Conv2d(3, 3, 3)
        
        n =6

        self.conv_layer_1 = nn.Conv2d(3, self.net_width[0], 3)
        self.max_pooling_2 = nn.MaxPool2d(2, stride=2)

        if self.k>=4:
          n = 5
          self.conv_layer_2 = nn.Conv2d(self.net_width[0], 3, 3)
          self.conv_layer_3 = nn.Conv2d(3, self.net_width[1], 3)
        else:
          self.conv_layer_3 = nn.Conv2d(self.net_width[0], self.net_width[1], 3) 

        self.max_pooling_4 = nn.MaxPool2d(2, stride=2)

        if self.k>=5:
          n = 3
          self.conv_layer_4 = nn.Conv2d(self.net_width[1], self.net_width[1], 3)

        self.fc_5 = nn.Linear(self.net_width[1] * n * n, 784)
        if p_dropout >= 0:
          self.dropout = nn.Dropout(p_dropout)
        else:
          self.dropout = None

        self.fc_6 = nn.Linear(784, 10)

    def random_init_specific_layer(self, layer_to_init, std):
        layer_to_init.weight.data = layer_to_init.weight.data.normal_(mean=0.0, std=std).to(torch.device("cuda:0"))
        layer_to_init.bias.data = nn.init.zeros_(layer_to_init.bias.data).to(torch.device("cuda:0"))

    def random_init(self, std=1.0):
        if self.k >= 3:
            self.random_init_specific_layer(self.conv_layer_0, std)

        self.random_init_specific_layer(self.conv_layer_1, std)

        if self.k >= 4:
            self.random_init_specific_layer(self.conv_layer_2, std)

        self.random_init_specific_layer(self.conv_layer_3, std)

        if self.k == 5:
            self.random_init_specific_layer(self.conv_layer_4, std)

        self.random_init_specific_layer(self.fc_5, std)
        self.random_init_specific_layer(self.fc_6, std)
      
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

        hidden_activation_0 = x
        if self.k >= 3:
          hidden_activation_0 = F.relu(self.conv_layer_0(x))

        hidden_activation_1 = F.relu(self.conv_layer_1(hidden_activation_0))
        hidden_activation_2 = self.max_pooling_2(hidden_activation_1)

        hidden_activation_3 = hidden_activation_2
        if self.k >= 4:
          hidden_activation_3 = F.relu(self.conv_layer_2(hidden_activation_2))

        hidden_activation_4 = F.relu(self.conv_layer_3(hidden_activation_3))
        hidden_activation_5 = self.max_pooling_4(hidden_activation_4)

        hidden_activation_6 = hidden_activation_5
        if self.k == 5:
          hidden_activation_6 = F.relu(self.conv_layer_4(hidden_activation_5))

        n = 6 if self.k in (2,3) else (3 if self.k==5 else 5)
        hidden_activation_7 = self.fc_5(hidden_activation_6.reshape(-1, self.net_width[1] * n * n))
        if self.dropout:
          hidden_activation_7 = self.dropout(hidden_activation_7)

        output = self.fc_6(hidden_activation_7)
        return output