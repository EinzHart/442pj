import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

class basicNet(nn.Module):
    def __init__(self):
        super().__init__()
        # layers
        # conv layers
        self.conv1 = nn.Conv2d(1, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 3, 1)

    def forward(self,input):
        temp1 = self.conv1(input)
        temp2 = self.conv2(temp1)

        return temp2
