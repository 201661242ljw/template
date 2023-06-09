# import torch
# import torch.nn as nn
#
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResBlock, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * 4)
#
#         self.downsample = None
#         if stride != 1 or in_channels != out_channels * 4:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * 4),
#             )
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = nn.ReLU(inplace=True)(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = nn.ReLU(inplace=True)(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = nn.ReLU(inplace=True)(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000):
#         super(ResNet, self).__init__()
#
#         self.in_channels = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         layers = []
#
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels * block.expansion
#
#         for i in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = nn.ReLU(inplace=True)(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x
#
#
# class MyNet(nn.Module):
#     def __init__(self, layers, block=ResBlock):
#         super(MyNet, self).__init__()
#
#         self.in_channels = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, 6)
#
#     def _make_layer(self, block, out_channels, blocks, stride=1):
#         layers = []
#
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels * block.expansion
#
#         for i in range(1, blocks):
#             layers.append(block(self.in_channels, out_channels))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = nn.ReLU(inplace=True)(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#
#         return x


import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#
#         self.in_channels = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)
#
#     def make_layer(self, block, out_channels, num_blocks, stride):
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride))
#         self.in_channels = out_channels
#         for i in range(num_blocks - 1):
#             layers.append(block(out_channels, out_channels, stride=1))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#
#         return out
#
#
# def ResNet18(num_classes=10):
#     return ResNet(ResBlock, [2, 2, 2, 2], num_classes=num_classes)
#
#
# def ResNet34(num_classes=10):
#     return ResNet(ResBlock, [3, 4, 6, 3], num_classes=num_classes)


class MyNet(nn.Module):
    def __init__(self, num_blocks, block=ResBlock, num_classes=6):
        super(MyNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        )

        self.final = nn.Conv2d(256, 3, 1, 1)


    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.deconv(out)
        out = self.final(out)
        # out = self.avg_pool(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc(out)
        #
        out = self.sigmoid(out)

        return out
