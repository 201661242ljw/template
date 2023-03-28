import os.path
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 定义自己的数据集
class MyDataset(Dataset):
    def __init__(self, data_path, sigma=2):

        self.sigma = sigma


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

    def get_Heatmap(self):
        joints_3d = self.points
        num_joints = joints_3d.shape[0]
        image_size = np.array([self.width, self.height], dtype=np.float32)
        W, H = self.width // 4, self.height//4
        target_weight = np.ones((num_joints), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)
        # 3-sigma rule
        tmp_size = self.sigma * 3

        for joint_id in range(num_joints):
            if (joints_3d[joint_id][0] + joints_3d[joint_id][1]) == 0:
                target_weight[joint_id] = 0

            feat_stride = image_size / [W, H]
            mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0

            if target_weight[joint_id] > 0.5:
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, None]
                x0 = y0 = size // 2
                # The gaussian is not normalized,
                # we want the center value to equal 1
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], W)
                img_y = max(0, ul[1]), min(br[1], H)

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        
        return target, target_weight


    def __getitem__(self, idx):
        img_path = self.datas[idx].split()[0]
        label = self.datas[idx].split()[1:]

        img_path = os.path.join(self.project_dir, r"datasets/images", img_path)

        # 加载图像
        img = Image.open(img_path)
        self.width, self.height = img.size
        # 对图像进行预处理，并将其转换为张量
        img_data = self.transform(img)

        # 将字符串转换为浮点数，并将它们存储在列表中
        int_list = [int(x) for x in label]
        
        self.points = np.array(int_list, dtype=np.int).reshape((-1, 2))

        target, target_weight = self.get_Heatmap()


        # self.points = {"start": float_list[:2],
        #           "mediun": float_list[2:4],
        #           "end": float_list[4:]}

        # # 将列表转换为张量
        # label = torch.Tensor(float_list)
        # 
        # label = torch.reshape(label, (3, 2))
        # 
        # label[:, 0] /= width
        # label[:, 1] /= height

        return img_data, target, target_weight


def my_dataset_collate(batch):
    pass
