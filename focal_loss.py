# Know more about focal loss https://arxiv.org/abs/1708.02002
# This scripts is wrote by reference to https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py
# A little revisement
# This script is for gpu version, you can change it for a cpu version

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(target, classes):
    mask = torch.rand(target.size(0),classes).zero_()
    mask.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
    mask = torch.autograd.Variable(mask.cuda())

    return mask


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        # input, target is instances of autograd.Variable on cuda 
        # input.size()  ([batch_size, classes])
        # target.size() ([batch_size])
        y = one_hot(target, input.size(1))

        logit = F.softmax(input)
        # logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * ((1 - logit) ** self.gamma) / float(input.size(0)) # focal loss

        return loss.sum()
