import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
label_name = {}


class DataSet(Dataset):
    def __init__(self, data_path, label_name,transform=None):  # 除了这两个参数之外，还可以设置其它参数
        self.label_name = label_name
        # self.data_info = get_info_list(data_path)
        self.data_info = get_info(data_path, label_name)
        self.transform = transform

    def __getitem__(self, index):
        label, img_path = self.data_info[index]
        pil_img = Image.open(img_path).convert('RGB')  # 读数据
        pil_img = transforms.Resize((224, 224))(pil_img)
        plt.figure(0)
        # ax = plt.subplot(121)
        # ax.set_title('original image')
        # ax.imshow(pil_img)
        if self.transform:
            img = self.transform(pil_img)

        # re_img = transforms.Resize((32, 32))(pil_img)
        # img = transforms.ToTensor()(re_img)  # PIL转张量

        return img, label

    def __len__(self):
        return len(self.data_info)


def get_info(data_path, label_name):
    data_info = list()
    for root_dir, sub_dirs, _ in os.walk(data_path):
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))
            img_names = list(filter(lambda x: x.endswith('.jpg'), file_names))
            for i in range(len(img_names)):
                img_path = os.path.join(root_dir, sub_dir, img_names[i])
                img_label = label_name[sub_dir]
                data_info.append((img_label, img_path))

    return data_info


def get_info_list(list_path):
    data_info = list()
    with open(list_path, mode='r') as f:
        lines = f.readlines()
        for j in range(len(lines)):
            img_label = int(lines[j].split(',')[0])
            img_path = lines[j].split(',')[1].replace('\n', '')
            data_info.append((img_label, img_path))
    return data_info


if __name__ == '__main__':

    # label_name = {'ants': 0, 'bees': 1}
    # train_list_path = os.path.join('old_data', 'train_set.txt')
    # val_list_path = os.path.join('old_data', 'val_set.txt')
    # test_list_path = os.path.join('old_data', 'test_set.txt')
    # train_set = DataSet(data_path=train_list_path, label_name=label_name)
    # val_set = DataSet(data_path=val_list_path, label_name=label_name)
    # test_set = DataSet(data_path=test_list_path, label_name=label_name)

    label_name = {'unmasking': 0, 'masking': 1}
    train_set_path = os.path.join('../data', 'train_set')
    val_set_path = os.path.join('../data', 'val_set')
    test_set_path = os.path.join('../data', 'test_set')
    train_set = DataSet(data_path=train_set_path, label_name=label_name)
    val_set = DataSet(data_path=val_set_path, label_name=label_name)
    test_set = DataSet(data_path=test_set_path, label_name=label_name)

    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)


    for i, data in enumerate(train_loader):
        inputs, labels = data
        print(labels)


