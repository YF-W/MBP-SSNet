from torch import nn
# from models.unet_parts import OutConv
# from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
# from models.layers import CBAM
import torch
import torch.nn.functional as F



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


class ChannelAttention1(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1, self).__init__()
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


class DepthwiseSeparableConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv1, self).__init__()
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
        self.DW=DepthwiseSeparableConv1(in_channels,in_channels)
        self.A1=ChannelAttention1(in_channels)
        self.A2 = ChannelAttention(out_channels)
        self.conv=Conv(in_channels*2,out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(in_channels,in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x_1 = self.A1(x)
        x_2 = self.DCN(self.DCS(x)+x)
        x_2_p=torch.cat((x_2, x_1), dim=1)
        x_3 = self.conv(torch.cat((self.DW(x), x), dim=1))
        x_out=x_3+x_2_p

        return x_out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # 对tensor进行扩充，扩充的方向分别是左、右、上、下。
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class SmaAt_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(SmaAt_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DetailMega1(64, 128)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DetailMega1(128, 256)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DetailMega1(256, 512)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


# x=torch.randn(6,4,128,128)
# model=SmaAt_UNet(4,1)
# preds=model(x)
# print(preds.shape)
# print(x.shape)
