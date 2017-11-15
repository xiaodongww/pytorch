# This script is used is calssificaiton problems, it change the label Variable to one-hot Variable
# target(label) is an instanc of autograd.Variable on cuda  target.size = ([batch_size])
# classes is the num of classes

def one_hot(target, classes):
    mask = torch.rand(target.size(0),classes).zero_()
    mask.scatter_(1, target.view(target.size(0),1).data.cpu(), 1)
    mask = torch.autograd.Variable(mask.cuda())
    
    return mask
