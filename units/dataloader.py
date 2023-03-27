import os.path
import os
import torch
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 定义自己的数据集
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.datas = open(data_path, "r", encoding="utf-8").readlines()  # 加载所有图片的路径和标签信息
        self.transform = transforms.Compose([transforms.ToTensor()])

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取当前项目的根目录
        self.project_dir = os.path.dirname(current_dir)

        # 定义预处理函数，将图像转换为张量
        self.transform = transforms.Compose([
            # transforms.Resize(256),  # 调整图像大小为 256x256
            # transforms.CenterCrop(224),  # 中心裁剪 224x224
            transforms.ToTensor()  # 将图像转换为张量，并归一化到 [0,1] 范围
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        img_path = self.datas[idx].split()[0]
        label = self.datas[idx].split()[1:]

        img_path = os.path.join(self.project_dir, r"datasets/images", img_path)

        # 加载图像
        img = Image.open(img_path)
        width, height = img.size
        # 对图像进行预处理，并将其转换为张量
        img_data = self.transform(img)

        # 将字符串转换为浮点数，并将它们存储在列表中
        float_list = [float(x) for x in label]

        # 将列表转换为张量
        label = torch.Tensor(float_list)

        label = torch.reshape(label, (3, 2))

        label[:, 0] /= width
        label[:, 1] /= height

        return img_data, label


def my_dataset_collate(batch):
    pass
