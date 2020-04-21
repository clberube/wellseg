# @Author: charles
# @Date:   2020-04-21T10:13:31-04:00
# @Last modified by:   charles
# @Last modified time: 2020-04-21T10:50:42-04:00


import torch
from torch import nn
import torch.nn.functional as F

#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(784, 32)
#
#     def forward(self, x):
#         return F.sigmoid(self.fc1(x))
#
#
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.fc1 = nn.Linear(32, 784)
#
#     def forward(self, x):
#         return F.sigmoid(self.fc1(x))


class BalancedAE(nn.Module):
    def __init__(self):
        super(BalancedAE, self).__init__()
        self.encoder = nn.Parameter(torch.rand(784, 32))

    def forward(self, x):
        x = torch.sigmoid(torch.mm(self.encoder, x))
        x = torch.sigmoid(torch.mm(x, torch.transpose(self.encoder, 0, 1)))
        return x


class TiedAutoEncoderFunctional(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.param = nn.Parameter(torch.randn(out, inp))

    def forward(self, input):
        encoded_feats = F.linear(input, self.param)
        reconstructed_output = F.linear(encoded_feats, self.param.t())
        return encoded_feats, reconstructed_output


class TiedAutoEncoder(nn.Module):
    def __init__(self, inp, out, weight):
        super().__init__()
        self.encoder = nn.Linear(inp, out, bias=False)

    def forward(self, input):
        encoded_feats = self.encoder(input)
        reconstructed_output = F.linear(encoded_feats, self.encoder.weight.t())
        return encoded_feats, reconstructed_output
