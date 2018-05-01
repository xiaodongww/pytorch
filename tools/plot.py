def plot_aug(loader):
    import torchvision.transforms as transforms
    imgtensor, *_ = next(iter(loader))
    toimg = transforms.ToPILImage('RGB')
    img = toimg(imgtensor.squeeze())
    img.show()
