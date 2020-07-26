import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def transform_inverse(img, transform):
    # 将tensor转换成pil数据
    if 'Normalize' in str(transform):
        Normalize_trans = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        m = torch.tensor(Normalize_trans[0].mean, dtype=img.dtype, device=img.device)
        s = torch.tensor(Normalize_trans[0].std, dtype=img.dtype, device=img.device)
        img.mul_(torch.reshape(s, [-1, 1, 1])).add_(torch.reshape(m, [-1, 1, 1]))       # 需要调整形状才能通道对应
    img = torch.transpose(img, dim0=0, dim1=1)                                          # C H W ->H C W
    img = torch.transpose(img, dim0=1, dim1=2)                                          # H C W ->H W C

    img = np.array(img)*255                                                             # 去归一化
    # print(img)
    if img.shape[2] == 3:
        img = Image.fromarray(img.astype('uint8')).convert('RGB')                       # 转换成PIL RGB图像
    elif img.shape[2] == 1:
        img = Image.fromarray(img.astype('uint8').squeeze())                           # (1, H, W)->(H, W)
    else:
        print('Invalid img format')
    return img



