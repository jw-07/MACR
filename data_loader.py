# dataloader for mean teacher for s2r (simulation to real)

import torch
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


class MTData(Dataset):
    def __init__(self, image_dir, simulation=True, transform=None):
        self.image_dir = image_dir  # Folder path
        self.simulation = simulation  # Determine whether to read the simulated image
        self.alldata = os.listdir(self.image_dir)

        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.alldata[idx])
        img = Image.open(img_path).convert('RGB')
        img_fake_path = self.renamestr(img_path)
        img_fake = Image.open(img_fake_path).convert('RGB')
        label = self.alldata[idx].split('_')[0]
        classes = {'2s1': 0, 'bmp2': 1, 'btr70': 2, 'm1': 3, 'm2': 4, 'm35': 5, 'm60': 6, 'm548': 7, 't72': 8,
                   'zsu23': 9} # sample
        label = classes[label]
        if self.transform is not None:
            img = self.transform(img)
            img_fake = self.transform(img_fake)
        # return img, img_fake, label, img_path, img_fake_path
        return img, img_fake, label

    def __len__(self):
        return len(self.alldata)

    def renamestr(self, str1):
        if self.simulation:
            str2 = str1.replace('simulation', 'simulation_fake')  # simulation_fake path
        else:
            str2 = str1.replace('realdata', 'realdata_fake')  # realdata_fake path
        return str2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    label_dir = 'data/realdata'
    label_dir1 = 'data/simulation'
    data_transforms = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    dataset = MTData(label_dir, simulation=False, transform=data_transforms)
    dataset1 = MTData(label_dir1, simulation=True, transform=data_transforms)
    print(len(dataset))
    # print(dataset[0][3], dataset[0][4])
    print(dataset[0][0].shape, dataset[0][1].shape)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    loader1 = DataLoader(dataset1, batch_size=1, shuffle=False)


    def imshow(img):
        # img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # (a, b, c, d, e), (a1, b1, c1, d1, e1) = next(iter(zip(loader, loader1)))
    # # print(c,d,e)
    # print(c1, d1, e1)

    a, b, c = next(iter(loader1))
    a = make_grid(a, 5, 20)
    # save_image(a, 'out.jpg')
    imshow(a)
    b = make_grid(b, 5, 20)
    # save_image(b, 'out1.jpg')
    imshow(b)

