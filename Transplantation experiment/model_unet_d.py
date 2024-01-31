import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
import cv2 as cv
import warnings

warnings.filterwarnings("ignore")


class ConvF(nn.Module):
    # 初始化
    def __init__(self, in_channels, out_channels):
        super(ConvF, self).__init__()
        # 这里执行的是双卷积的初始化操作，用Sequential这个序列把他保存起来
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),  # 第一次卷积，卷积核3*3
            nn.BatchNorm2d(out_channels),  # 批标准化，用于放置梯度爆炸和梯度消失
            nn.GELU(),  # 激活函数，其中inplace参数用于检查是否进行覆盖运算

        )

    # 按序，将初始化的组件，进行拼装。这里不需要拼装，因为初始化的过程中已经把顺序定好了
    def forward(self, x):
        return self.conv(x)


class ConvN(nn.Module):
    # 初始化
    def __init__(self, in_channels, out_channels):
        super(ConvN, self).__init__()
        # 这里执行的是双卷积的初始化操作，用Sequential这个序列把他保存起来
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 9, 1, 4, bias=False),  # 第一次卷积，卷积核3*3
            nn.BatchNorm2d(out_channels),  # 批标准化，用于放置梯度爆炸和梯度消失
            nn.GELU(), # 激活函数，其中inplace参数用于检查是否进行覆盖运算

        )

    # 按序，将初始化的组件，进行拼装。这里不需要拼装，因为初始化的过程中已经把顺序定好了
    def forward(self, x):
        return self.conv(x)


class ConvE(nn.Module):
    # 初始化
    def __init__(self, in_channels, out_channels):
        super(ConvE, self).__init__()
        # 这里执行的是双卷积的初始化操作，用Sequential这个序列把他保存起来
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 11, 1, 5, bias=False),  # 第一次卷积，卷积核3*3
            nn.BatchNorm2d(out_channels),  # 批标准化，用于放置梯度爆炸和梯度消失
            nn.GELU(),  # 激活函数，其中inplace参数用于检查是否进行覆盖运算

        )

    # 按序，将初始化的组件，进行拼装。这里不需要拼装，因为初始化的过程中已经把顺序定好了
    def forward(self, x):
        return self.conv(x)


class ConvS(nn.Module):
    # 初始化
    def __init__(self, in_channels, out_channels):
        super(ConvS, self).__init__()
        # 这里执行的是双卷积的初始化操作，用Sequential这个序列把他保存起来
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7,1,3, bias=False),  # 第一次卷积，卷积核3*3
            nn.BatchNorm2d(out_channels),  # 批标准化，用于放置梯度爆炸和梯度消失
            nn.GELU(), # 激活函数，其中inplace参数用于检查是否进行覆盖运算

        )

    # 按序，将初始化的组件，进行拼装。这里不需要拼装，因为初始化的过程中已经把顺序定好了
    def forward(self, x):
        return self.conv(x)


import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层
        self.sep_conv = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes,
                                  bias=False)  # 深度可分离卷积
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 =   nn.GELU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sep_conv_out = self.sep_conv(x)  # 深度可分离卷积分支
        sep_conv_out = F.relu(sep_conv_out)

        sep_conv_out = self.avg_pool(sep_conv_out)  # 全局平均池化之前应用深度可分离卷积
        avg_out = self.fc2(self.relu1(self.fc1(sep_conv_out)))  # 全局平均池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(sep_conv_out)))) # 最大池化分支

        soft_x = F.softmax(x, dim=1)
        x = (x+ avg_out) * self.sigmoid(max_out )  # 使用加和后的注意力权重
        x = x * soft_x
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu =nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class Conv(nn.Module):
    # 初始化
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        # 这里执行的是双卷积的初始化操作，用Sequential这个序列把他保存起来
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),  # 第一次卷积，卷积核3*3
            nn.BatchNorm2d(out_channels),  # 批标准化，用于放置梯度爆炸和梯度消失
            # nn.ReLU(inplace=True),  # 激活函数，其中inplace参数用于检查是否进行覆盖运算

        )

    # 按序，将初始化的组件，进行拼装。这里不需要拼装，因为初始化的过程中已经把顺序定好了
    def forward(self, x):
        return self.conv(x)


class DetailMega1(nn.Module):
    # 初始化
    def __init__(self, in_channels, out_channels):  # x上一层图像，y：横向编码的连接结果
        super(DetailMega1, self).__init__()
        # 这里执行的是双卷积的初始化操作，用Sequential这个序列把他保存起来
        self.DCF=ConvF(in_channels,in_channels)  # 5.1.2
        self.DCS = ConvS(in_channels, in_channels)  # 7.1.3
        self.DCN = ConvN(in_channels, in_channels)  # 9.1.3
        self.DCE = ConvE(in_channels, in_channels)  # 9.1.3
        self.Dc=DoubleConv(in_channels,in_channels)
        self.DW=DepthwiseSeparableConv(in_channels,in_channels)
        self.A1=ChannelAttention(in_channels)
        self.A2 = ChannelAttention(out_channels)
        self.conv=Conv(in_channels,out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(in_channels,in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x_1 = self.A1(x)
        x_2 = self.DCN(self.DCS(x)+x)
        x_2_p=torch.cat((x_2, x_1), dim=1)
        x_3 = torch.cat((self.DW(x), x), dim=1)
        x_out=x_3+x_2_p

        return x_out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(  # nn.Sequential 有序容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace 是否进行覆盖运算
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# kernel = [[0, -1, 0],
#           [-1, 5, -1],
#           [0, -1, 0]]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# kernel = torch.FloatTensor(kernel).expand(8,512,3,3)
# weight = torch.nn.Parameter(data=kernel, requires_grad=False).to(device=DEVICE)


# class GaussianBlurConv(nn.Module):
#     def __init__(self, channels):
#         super(GaussianBlurConv, self).__init__()
#         self.channels = channels
#         kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
#                   [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
#                   [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
#                   [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
#                   [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
#         kernel = np.repeat(kernel, self.channels, axis=0)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#
#     def __call__(self, x):
#         x = nn.Conv2d(x.unsqueeze(0), self.weight, padding=2, groups=self.channels)
#         return x


def get_kernel():
    """
    See https://setosa.io/ev/image-kernels/
    """

    # k1:blur k2:outline k3:sharpen

    k1 = np.array([[0.0625, 0.125, 0.0625],
                   [0.125, 0.25, 0.125],
                   [0.0625, 0.125, 0.0625]])

    # Sharpening Spatial Kernel, used in paper
    k2 = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

    k3 = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

    return k1, k2, k3


def build_sharp_blocks(layer):
    """
    Sharp Blocks
    """
    # Get number of channels in the feature
    in_channels = layer.shape[1]
    # Get kernel
    _, w, _ = get_kernel()
    # Change dimension
    w = np.expand_dims(w, axis=0)
    # Repeat filter by in_channels times to get (H, W, in_channels)
    w = np.repeat(w, in_channels, axis=0)
    # Expand dimension
    w = np.expand_dims(w, axis=0)
    return torch.FloatTensor(w)


class UNET(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],featuress=[ 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()  # 将多个Module加入list，但不存在实质性顺序，参考python的list
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in featuress:
            self.downs.append(DetailMega1(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.D=DoubleConv(3,64)
        self.DetailMega64_128=DetailMega1(64,128)
        self.DetailMega128_256 = DetailMega1(128,256)
        self.DetailMega256_512 = DetailMega1(256,512)

    def forward(self, x):
        skip_connections = []

        # decoder part
        x=self.D(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.DetailMega64_128(x)
        skip_connections.append(x)
        x= self.pool(x)
        x = self.DetailMega128_256(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.DetailMega256_512(x)
        skip_connections.append(x)
        x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # encoder part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# x = torch.randn(4, 3,512,512)
# model = UNET(in_channels=3,out_channels=1)
# preds = model(x)
# print(x.shape)
# print(preds.shape)