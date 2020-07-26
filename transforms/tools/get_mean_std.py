import torch
import random
import time
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np


def get_mean_std(dataset, ratio=1):
    dataloader = DataLoader(dataset, batch_size=int(ratio*len(dataset)),
                            shuffle=True, num_workers=0)
    data = iter(dataloader).next()[0]  # 一个batch的数据

    mean = np.mean(data.numpy(), axis=(0, 2, 3))
    std = np.std(data.numpy(), axis=(0, 2, 3))
    return mean, std


def get_std_mean(dataset, ratio=1):
    data_x = dataset.data
    data_x = torch.transpose(torch.from_numpy(data_x), dim0=1, dim1=3)  # (50000,32,32)->(50000,3,32,32)
    data_num = len(data_x)
    idx = list(range(data_num))
    random.shuffle(idx)  # 产生随机索引
    data_selected = data_x[idx[0:int(ratio * data_num)]]
    mean = np.mean(data_selected.numpy(), axis=(0, 2, 3)) / 255
    std = np.std(data_selected.numpy(), axis=(0, 2, 3)) / 255
    return mean, std


if __name__ == '__main__':
    train_dataset = datasets.CIFAR10('./data',
                                     train=True, download=False,
                                     transform=transforms.ToTensor())

    test_dataset = datasets.CIFAR10('./data',
                                    train=False, download=False,
                                    transform=transforms.ToTensor())
    time0 = time.time()

    train_mean, train_std = get_std_mean(train_dataset)
    test_mean, test_std = get_std_mean(test_dataset)
    time1 = time.time()
    time = time1 - time0
    print(time)
    print(train_mean, train_std)
    print(test_mean, test_std)
