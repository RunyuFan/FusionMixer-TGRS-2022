import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.io as io
import torchvision
from torchvision import transforms as T
import numpy as np
def MyLoader(path,type):
    if type=='img':
        return Image.open(path).convert('RGB')
    elif type=='vector':
        return np.load(path, allow_pickle=True)
    elif type=='msi':
        return io.loadmat(path)['msi']


class Mydataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None, loader=MyLoader):
        X = np.load(X)
        y = np.load(y)
        # print(X.shape, y.shape, )
        file=[]
        for idx in range(X.shape[0]):
            # line=line.strip('\n')
            # line=line.rstrip()
            # words=line.split()
            # print()
            file.append((X[idx, 0, :, :, :], X[idx, 1, :, :, :], X[idx, 2, :, :, :], X[idx, 3, :, :, :], X[idx, 4, :, :, :], int(y[idx]))) # 路径1 路径2 路径3 路径4 路径5 标签


        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


    def __getitem__(self, index):

        hrs, sv0, sv1, sv2, sv3, label = self.file[index]
        # print(hrs.shape, sv0.shape, sv3.shape)

        # hrs_f=self.loader(hrs,type='img')
        # sv_f=self.loader(sv,type='img')
        # msi=self.loader(lrs,type='msi')
        # checkin_f = np.array(self.loader(checkin, type='vector'))
        # print(checkin_f)

        if self.transform is not None:
            hrs_f = self.transform(torch.from_numpy(hrs).permute(2, 0, 1).float())
            sv_f0 = self.transform(torch.from_numpy(sv0).permute(2, 0, 1).float())
            sv_f1 = self.transform(torch.from_numpy(sv1).permute(2, 0, 1).float())
            sv_f2 = self.transform(torch.from_numpy(sv2).permute(2, 0, 1).float())
            sv_f3 = self.transform(torch.from_numpy(sv3).permute(2, 0, 1).float())
            # msi=torch.from_numpy(msi*1.0)[4:, :, :]
            # # msi = self.transform(msi)
            # checkin_f=torch.from_numpy(checkin_f).reshape(120, 1)*1.0
        # print(hrs_f.shape, msi.shape,hpi_f.shape,sv_f.shape,checkin_f.shape,floor_f.shape)

        return hrs_f, sv_f0, sv_f1, sv_f2, sv_f3, label

    def __len__(self):
        return len(self.file)


if __name__ == "__main__":
    test_transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256))])
    test_dataset=Mydataset(X=r'J:\\data\\train_X.npy', y=r'J:\\data\\train_Y.npy', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False,pin_memory=True)
    for step, (hrs_f, sv_f0, sv_f1, sv_f2, sv_f3, label) in enumerate(test_loader):
        print(hrs_f.shape, sv_f0.shape, sv_f3.shape, label.shape)
