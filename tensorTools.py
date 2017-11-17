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
