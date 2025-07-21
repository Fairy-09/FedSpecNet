import torch.nn as nn
import torch


class LSTM(nn.Module):
    # def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=1):
    def __init__(self, input_size=1, hidden_size=4, output_size=1, num_layer=1,bidirectional=True):
        super(LSTM, self).__init__()       ##调用父类的构造函数进行初始化
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer,bidirectional=bidirectional)   ##布尔值，指示是否使用双向LSTM
        self.layer2 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        ##第二层做了修改的
        ##第一层为LSTM，第二层为全连接层（线性），用于从LSTM的输出生成最终输出

    def forward(self, x):
        x, _ = self.layer1(x)
        x = torch.relu(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x[:,-1,:]