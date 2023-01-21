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
    def __init__(self,txt,transform=None, target_transform=None, loader=MyLoader):
        with open(txt,'r') as fh:
            file=[]
            for line in fh:
                line=line.strip('\n')
                line=line.rstrip()
                words=line.split()
                file.append((words[0],words[1], int(words[-1]))) # 路径1 路径2 路径3 路径4 路径5 标签


        self.file=file
        self.transform=transform
        self.target_transform=target_transform
        self.loader=loader


    def __getitem__(self,index):

        hrs,sv,label=self.file[index]

        hrs_f=self.loader(hrs,type='img')
        sv_f=self.loader(sv,type='img')
        # msi=self.loader(lrs,type='msi')
        # checkin_f = np.array(self.loader(checkin, type='vector'))
        # print(checkin_f)

        if self.transform is not None:
            hrs_f=self.transform(hrs_f)
            sv_f=self.transform(sv_f)
            # msi=torch.from_numpy(msi*1.0)[4:, :, :]
            # # msi = self.transform(msi)
            # checkin_f=torch.from_numpy(checkin_f).reshape(120, 1)*1.0
        # print(hrs_f.shape, msi.shape,hpi_f.shape,sv_f.shape,checkin_f.shape,floor_f.shape)

        return hrs_f, sv_f, label

    def __len__(self):
        return len(self.file)


if __name__ == "__main__":
    test_transform=torchvision.transforms.Compose([# torchvision.transforms.Resize((256,256)),
                                            torchvision.transforms.ToTensor()])
    train_txt='data\\train_2_10.txt'
    val_txt='data\\val_2_10.txt'
    test_txt='data\\test_6_10.txt'

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator

    train_dataset=Mydataset(txt=train_txt,transform=test_transform)
    val_dataset=Mydataset(txt=val_txt,transform=test_transform)
    test_dataset=Mydataset(txt=test_txt,transform=test_transform)

    width, lenth = [], []
    for hrs_f, sv_f, label in train_dataset:
        # print(hrs_f.shape)
        width.append(hrs_f.shape[1])
        lenth.append(hrs_f.shape[2])
    for hrs_f, sv_f, label in test_dataset:
        # print(hrs_f.shape)
        width.append(hrs_f.shape[1])
        lenth.append(hrs_f.shape[2])
    for hrs_f, sv_f, label in val_dataset:
        # print(hrs_f.shape)
        width.append(hrs_f.shape[1])
        lenth.append(hrs_f.shape[2])


    # print(len(width), len(lenth), np.mean(width), np.mean(lenth), max(width), max(lenth), min(width), min(lenth))  # 3236 3236 350.32725587144625 383.3142768850433 1100 1242 87 91
    plt.figure(figsize=(6, 6))
    # plt.subplot(121)
    # plt.scatter(width, lenth, s=5, c='r')
    # # plt.title('The distribution of the size of parcels')
    # # plt.xlabel('The width of parcels', fontsize=150)
    # # plt.ylabel('The length of parcels', fontsize=150)
    # # y_major_locator=MultipleLocator(100)
    # # ax.yaxis.set_major_locator(y_major_locator)
    # plt.yticks(range(0, 1400, 100))

    # plt.subplot(122)
    plt.boxplot((width, lenth),
            labels=('',''),
            medianprops={'color': 'red', 'linewidth': '1'},
            meanline=True,
            patch_artist=True,
            boxprops={'facecolor':'pink'},
            showmeans=True,
            showfliers=True)

    # plt.ylim((0, 1))
    # plt.xlim((0,5000))
    plt.grid(linestyle="--", alpha=0.3)
    # plt.xlabel('The width of parcels')
    # plt.ylabel('Value', fontsize=150)
    plt.yticks(range(0, 1400, 100))
    plt.savefig('box-2.png', format='png')

    # test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False,pin_memory=True)
    # for step,(x1,x2,label) in enumerate(test_loader):
    #     print(x1[:, :, 0, 0], x1.shape, x2.shape, label.shape)  # 0, 0, 0
