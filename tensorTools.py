# Tools for tensors

def mask(length):
    # return an upper triangular matrix
    mask = np.ones([length, length])
    mask = np.triu(mask)
    return torch.Tensor(mask)
# memory inefficient and the speed is low
# please use function pariwise_distances
def distanceMatrix(mtx1, mtx2):
    # dist_ij = Euclidean distance of mtx1[i] and mtx2[j]
    m = mtx1.size(0)
    p = mtx1.size(1)
    mmtx1 = torch.stack([mtx1]*m)
    mmtx2 = torch.stack([mtx2]*m).transpose(0, 1)
    dist = torch.sum((mmtx1 - mmtx2)**2, 2).squeeze()
    return dist

# source from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def replace(x, value):
    """
    Funtion: replace 0 with a specific value and set non zero elemets to zero 
    x is a matrix with elemetn 0 and non-zero element
    output[i,j] = value if x[i,j] equals to 0
    output[i,j] = 0 if x[i,j] is not equal to 0
    """
    output = (x == 0).float().mul(value)
