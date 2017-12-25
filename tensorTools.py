# Tools for tensors

def mask(length):
    # return an upper triangular matrix
    mask = np.ones([length, length])
    mask = np.triu(mask)
    return torch.Tensor(mask)

def distanceMatrix(mtx1, mtx2):
    # dist_ij = Euclidean distance of mtx1[i] and mtx2[j]
    m = mtx1.size(0)
    p = mtx1.size(1)
    mmtx1 = torch.stack([mtx1]*m)
    mmtx2 = torch.stack([mtx2]*m).transpose(0, 1)
    dist = torch.sum((mmtx1 - mmtx2)**2, 2).squeeze()
    return dist

def replace(x, value):
    """
    Funtion: replace 0 with a specific value and set non zero elemets to zero 
    x is a matrix with elemetn 0 and non-zero element
    output[i,j] = value if x[i,j] equals to 0
    output[i,j] = 0 if x[i,j] is not equal to 0
    """
    output = (x == 0).float().mul(value)
