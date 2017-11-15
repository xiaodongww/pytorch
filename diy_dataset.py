# This a file for you to load diy datasets
# To use this script, please organiize the imgs as files in the directory DATA_ROOT
# This script also allows load other features
# In DATA_ROOT, *_imgs is the directory for imgs, *.txt contains the labels and names, *_fea.json is the additional features
# The parameters for initialization is a little different from the official api


import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import json


defaullt_transform = transforms.Compose([transforms.ToTensor()])  
def default_loader(path, resize,transform): 
    img = Image.open(path).convert('RGB')
    img = img.resize((resize,resize))
    return transform(img)


class DIY_DATASET(data.Dataset):
    def __init__(self, root, train='train', transform=default_transform, Additional=False,resize=256):
        self.root = os.path.expanduser(root)

        if train not in ['train', 'val', 'testa', 'testb']:
            raise KeyError

        self.train = train
        self.transform = transform
        print('Loading {} data...'.format(self.train))
        self.imgf_path = os.path.join(self.root, self.train+'_imgs')
        self.names = []
        self.labels = []
        self.added_fea = []
        self.resize = int(resize)
        with open(os.path.join(self.root, self.train+'.txt')) as f:
            for line in f:
                name, label = line.strip().split(' ')
                self.names.append(name.strip()) 
                self.labels.append(int(label.strip()))   
        self.length = len(self.names)
        self.Additional = Additional
        if Additional:
            with open(os.path.join(self.root, self.train+'_added_fea.txt')) as f:
                added_fea_json = json.load(f)
                for key in added_fea_json.keys():
                    self.added_fea.append(torch.Tensor(added_fea_json[key]))



    def __getitem__(self, index):
        img_path = os.path.join(self.imgf_path,self.names[index])
        img = default_loader(img_path,self.resize, self.transform)
        if not self.Additional:
            return img, self.labels[index]
        else:
            return img, self.labels[index], self.added_fea[index]

    def __len__(self):
        return self.length
    
    def getName(self):
        pass
#------------------------------------------usage start----------------------------------------------#
if __name__ == '__main__':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    crop_size = 224  
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomSizedCrop(crop_size), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    val_transform = transforms.Compose(
        [transforms.CenterCrop(crop_size), transforms.ToTensor(), transforms.Normalize(mean, std)])


    train_data = DIY_DATASET('DATA_ROOT', train='train', transform=train_transform, Additional=False, resize=256) 
    val_data = DIY_DATASET('DATA_ROOT', train='val', transform=val_transform, Additional=True, resize=256)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=24, shuffle=True,
                                               num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False,
                                              num_workers=4, pin_memory=True)

    data, target = next(iter(train_loader))
    data = torch.autograd.Variable(data.cuda())
    target = torch.autograd.Variable(target.cuda())

#------------------------------------------usage end----------------------------------------------#
