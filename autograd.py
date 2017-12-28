# optimize the weight manuallly by minus gradient
# pay attention to volatie and requires_grad
# See the original post through the following link
# https://discuss.pytorch.org/t/why-volatility-changed-after-operation-parameter-parameter-0-01-parameter-grad/11639

import torch
from torch.autograd import Variable
torch.manual_seed(1024)

input = Variable(torch.rand(100, 12))
parameter = Variable(torch.rand(100, 12), requires_grad=True, volatile=False)


for i in range(10):
    loss = ((input*parameter).abs()-1).abs().sum()
    loss.backward()
    parameter.data.copy_(parameter.data - (0.01 * parameter.grad.data))

# 1) Wrong: following code will make parameter a volatile variable because parameter.grad is volatile
# parameter = parameter - (0.01 * parameter.grad)

# 2) Wrong: after the operations below, there will be some problems for backward. 
# parameter will not receive backward. parameter.grad will becomes a None type
# parameter = parameter - (0.01 * Variable(parameter.grad.data, requires_grad=True, volatile=False))

# 3) right
# parameter.data.copy_(parameter.data - (0.01 * parameter.grad.data))
