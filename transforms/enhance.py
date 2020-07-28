import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tools.dataload import DataSet
from torchvision import transforms
from torch.utils.data import DataLoader
from tools.transform_inverse import transform_inverse
import tools.my_transforms as my_transforms

label_name = {'masking': 1, 'unmasking': 0}
BATCH_SIZE = 1
EPOCH = 6
mean = [0.581, 0.535, 0.514]
std = [0.299, 0.299, 0.304]

transform = transforms.Compose([

    # transforms.CenterCrop(447),
    # transforms.RandomCrop(224, padding=100, pad_if_needed=True, padding_mode='reflect'),
    # transforms.RandomResizedCrop(224),
    # transforms.FiveCrop(160),
    # transforms.TenCrop(160, vertical_flip=True),
    # transforms.Lambda(lambda pics: torch.stack([(transforms.ToTensor()(pic)) for pic in pics])),
    # transforms.RandomResizedCrop(224),

    # transforms.RandomVerticalFlip(0.5),
    # transforms.RandomHorizontalFlip(0.5),

    # transforms.RandomRotation(90, resample=2),
    # transforms.RandomRotation(90, resample=2, expand=True, fill=(0, 0, 255)),
    # transforms.RandomRotation(90, resample=2, center=(224, 0),expand=True),

    # transforms.Pad((30, 200, 100, 20), fill=200),
    # transforms.Pad((30, 200, 100, 20), fill=200, padding_mode='edge'),

    # transforms.ColorJitter(brightness=(1, 2)),
    # transforms.ColorJitter(contrast=(1, 2)),
    # transforms.ColorJitter(contrast=(0, 1)),
    # transforms.ColorJitter(saturation=(0, 1)),
    # transforms.ColorJitter(hue=0.5),

    # transforms.RandomGrayscale(0.5),

    # transforms.RandomAffine(degrees=60),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.5)),
    # transforms.RandomAffine(degrees=0, scale=(0.5, 1.5), fillcolor=(255, 0, 255)),
    # transforms.RandomAffine(degrees=0, shear=(40, 40), fillcolor=(0, 255, 255)),

    # transforms.RandomChoice( [transforms.RandomGrayscale(1), transforms.ColorJitter(hue=0.5),
    # transforms.RandomRotation(60, resample=2)]), transforms.RandomApply([transforms.Pad(50, fill=(100, 130, 110)),
    # transforms.RandomAffine(degrees=0, shear=40), transforms.RandomVerticalFlip(1)], 0.5),
    # transforms.RandomOrder([transforms.Pad(100, fill=(100, 130, 110)), transforms.CenterCrop(200)]),

    # my_transforms.AddSaltPepperNoise(0.2),
    my_transforms.AddGaussianNoise(mean=0, variance=1, amplitude=20),
    transforms.ToTensor(),

    # transforms.RandomErasing(value=(100, 130, 110)),
    # transforms.RandomErasing(scale=(0.1, 0.15), value='xyz', ratio=(1, 1)),

    transforms.Normalize(mean=mean, std=std),

])

if __name__ == '__main__':

    train_set_path = os.path.join('data', 'train_set')

    train_set = DataSet(data_path=train_set_path, label_name=label_name, transform=transform)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCH):
        img = iter(train_loader).next()[0]
        img = torch.squeeze(img, dim=0)
        print(img.shape)
        if len(img.shape) == 4:  # 多张图片组成的四维张量
            for i in range(len(img)):
                pil_img = transform_inverse(img[i], transform)
                ax1 = plt.subplot(3, 4, i + 2)
                ax1.set_title('processed image')
                ax1.imshow(pil_img)
        elif len(img.shape) == 3:
            pil_img = transform_inverse(img, transform)
            ax1 = plt.subplot(2, 3, epoch + 1)
            ax1.set_title('processed image')
            ax1.imshow(pil_img)

    plt.show()
