'''MLP in PyTorch.'''
import torch
import torch.nn as nn
import math
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = torch.tanh(self.fc1(x))
        out = self.fc2(out)
        # out = self.softmax(out)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
