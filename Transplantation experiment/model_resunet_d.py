import torch
import torch.nn as nn


class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

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


class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat((x, skip), dim=1)
        x = self.r(x)
        return x


class build_resunet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        """ Encoder 2 and 3 """
        self.r2 = DetailMega1(64, 128)
        self.r3 = DetailMega1(128, 256)

        """ Bridge """
        self.r4 = DetailMega1(256, 512)

        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)

        """ Output """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = torch.sigmoid

    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.pool(self.r2(skip1))
        skip3 = self.pool(self.r3(skip2))

        """ Bridge """
        b = self.pool(self.r4(skip3))

        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ output """
        output = self.output(d3)
        # output = self.sigmoid(output)

        return output

#
# if __name__ == "__main__":
#     inputs = torch.randn((4, 3, 256, 256))
#     model = build_resunet()
#     y = model(inputs)
#     print(y.shape)
