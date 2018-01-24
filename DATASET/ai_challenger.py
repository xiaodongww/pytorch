from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import skimage.io
import skimage.transform
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
caffe_root = '/home/wangwb/caffe/python'
sys.path.insert(0, caffe_root)
import caffe
from caffe.proto import caffe_pb2
import lmdb




class Ai_Challenger(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, train='train',
                 resize = 256, transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if train not in ['train', 'val', 'test_a', 'test_b']:
            raise KeyError

        self.train = train  # training set or test set

        if download:
            self.download()

        data_root = os.path.join(self.root, self.train+'_data')
        pkldb = os.path.join(data_root, 'data_key_{}.pkl'.format(resize))
        print ('Loading {} data...'.format(self.train))
        # now load the picked numpy arrays
        if self.train in ['train', 'val', 'test_a', 'test_b']:
            # self.db_path = os.path.join(self.root, 'train_data', 'lmdb', 'ai_challenger_{}_{}_lmdb'.format(self.train, resize))
            self.db_path = os.path.join(self.root, 'train_data', 'lmdb', 'ai_challenger_{}_{}_lmdb'.format(self.train, resize))
            self.env = lmdb.open(self.db_path, max_readers=1,
                readonly=True, lock=False, readahead=False, meminit=False)
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()['entries']
                print (self.length)
            if os.path.isfile(pkldb):
                with open(pkldb, 'rb') as f:
                    self.keys = pickle.load(f)
            else:
                with self.env.begin(write=False) as txn:
                    if resize == 256 or resize == 384:
                        self.keys = [key for key, _ in txn.cursor()]
                with open(pkldb, 'wb') as f:
                    pickle.dump(self.keys, f)
            print ('{} data done.'.format(self.train))

        else:
            pass
            

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            dtstr = cursor.get(self.keys[index])
        datum = caffe_pb2.Datum()
        datum.ParseFromString(dtstr)
        target = datum.label[0] # this caffe is multi-label version
        img = caffe.io.datum_to_array(datum)
        img = img[[2, 1, 0], :, :]
        img = np.transpose(img, (1, 2, 0))

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def _check_integrity(self):
        pass

    def download(self):
        pass
