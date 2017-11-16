# This script is a loss function for calssification
# When the max probability of a sample is larger than threshold, we set the its probability to be zero
# The purpose is to decay the effect of the easy samples and try to focus on the hard samples whose max probability is usually small

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(target, classes):
    mask = torch.rand(target.size(0),classes).zero_()
    mask.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
    mask = torch.autograd.Variable(mask.cuda())

    return mask

def truncate(input, threshold):
    threshold = -threshold
    output = -input
    nn_threshold = nn.Threshold(threshold, -1)
    output = nn_threshold(output)
    output = -output
    return output


class TruncatedLoss(nn.Module):

    def __init__(self, threshold=0.8):
        super(TruncatedLoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        y = one_hot(target, input.size(1))

        logit = F.softmax(input)
        logit = truncate(logit, self.threshold)
        loss = -1 * y * torch.log(logit) # cross entropy
        return loss.sum()/float(input.size(0))
