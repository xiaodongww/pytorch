import torch
from torch.utils.data.sampler import Sampler
import numpy as np
import random

class categoryRandomSampler(Sampler):
    def __init__(self, numBatchCategory, targets, batch_size):
        """
        This sampler will sample numBatchCategory categories in each batch.
        """
        self.targets = list(targets)
        self.batch_size = batch_size
        self.num_samples = len(targets)
        self.numBatchCategory = numBatchCategory
        num_categories = max(targets) + 1
        self.categoy_idxs = {}

        for i in range(num_categories):
            self.categoy_idxs[i] = []

        for i in range(self.num_samples):
            self.categoy_idxs[targets[i]].append(i)
       

    def __iter__(self):
        num_batches = self.num_samples//self.batch_size
        selected = []
        for i in range(num_batches):
            categories_selcted = np.random.randint(100, size=self.numBatchCategory)
            samplePool = []
            for j in categories_selcted:
                samplePool.extend(self.categoy_idxs[j])
            random.shuffle(samplePool)
            batch = samplePool[:self.batch_size]
            selected.extend(batch)

        return iter(torch.LongTensor(selected))

    def __len__(self):
        return self.num_samples
