import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models as resnet_model


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # for _ in range(depth):
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class encoder(nn.Module):
    # def __init__(self):
    #     super(encoder,self).__init__()

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.channelTrans = nn.Conv2d(in_channels=65, out_channels=512, kernel_size=1, padding=0)

    def forward(self, x):
        x_vit = x
        x_vit = self.to_patch_embedding(x_vit)
        b, n, _ = x_vit.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x_vit = torch.cat((cls_tokens, x_vit), dim=1)
        x_vit += self.pos_embedding[:, :(n + 1)]
        x_vit = self.dropout(x_vit)

        vit_layerInfo = []
        for i in range(4):
            x_vit = self.transformer(x_vit)
            vit_layerInfo.append(x_vit)

        return vit_layerInfo


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.conv(x)


class Conv_32(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Conv_32, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.conv(x)


class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DNET(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], out_channels=1):
        super(DNET, self).__init__()

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = encoder(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)
        self.finalconv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.DoubleConv3_64 = DoubleConv(3, 64)
        self.DoubleConv16_64 = DoubleConv(16, 64)
        self.DoubleConv8_16 = DoubleConv(8, 16)
        self.DoubleConv16_32 = DoubleConv(16, 32)
        self.DoubleConv16_16 = DoubleConv(16, 16)
        self.DoubleConv32_64 = DoubleConv(32, 64)
        self.DoubleConv64_64 = DoubleConv(64, 64)
        self.DoubleConv32_32 = DoubleConv(32, 32)
        self.DoubleConv64_128 = DoubleConv(64, 128)
        self.DoubleConv128_128 = DoubleConv(128, 128)
        self.DoubleConv128_256 = DoubleConv(128, 256)
        self.DoubleConv256_256 = DoubleConv(256, 256)
        self.DoubleConv256_512 = DoubleConv(256, 512)
        self.DoubleConv512_512 = DoubleConv(512, 512)
        self.DoubleConv512_1024 = DoubleConv(512, 1024)
        self.DoubleConv1024_2048 = DoubleConv(1024, 2048)
        self.DoubleConv1024_1024 = DoubleConv(1024, 1024)
        self.DoubleConv = DoubleConv(64, 128)
        self.up64_64 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up128_128 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.up256_256 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.up64_32 = UPConv(64, 32)
        self.up32_16 = UPConv(32, 16)
        self.up64_3 = UPConv(64, 3)
        self.up128_64 = UPConv(128, 64)
        self.up16_8 = UPConv(16, 8)
        self.up512_256 = UPConv(512, 256)
        self.up1024_512 = UPConv(1024, 512)
        self.up256_128 = UPConv(256, 128)
        self.final_conv = Conv_32(64, out_channels)
        self.conv1024_512 = Conv(1024, 512)
        self.conv128 = Conv(65 + 128, 128)
        self.conv256 = Conv(65 + 256, 256)
        self.conv512 = Conv(512 + 65, 512)
        self.conv64128_64 = Conv(64 + 128, 64)
        self.conv128256_128 = Conv(128 + 256, 128)
        self.conv384_512 = Conv(128 + 256, 512)
        self.conv64 = Conv(129, 64)
        self.conv512_256 = Conv(512, 256)
        self.conv128_64 = Conv(128, 64)
        self.conv256_128 = Conv(256, 128)
        self.conv192_64 = Conv(192, 128)

        self.vitLayer_UpConv = nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2)

    def forward(self, x):
        x_o1 = x
        x_o2 = x

        e0 = self.firstconv(x_o2)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e1 = self.encoder1(e0)  # torch.Size([4, 64,112
        e2 = self.encoder2(e1)  # torch.Size([4, 128,56
        e3 = self.encoder3(e2)  # torch.Size([4, 256,28
        e4 = self.encoder4(e3)  # torch.Size([4, 512, 14

        vit_layerInfo = self.encoder(x_o1)
        vit_layerInfo = vit_layerInfo[::-1]

        v1 = vit_layerInfo[0].view(4, 65, 14, 14)  # 28
        v1_1 = self.vitLayer_UpConv(v1)

        v2 = vit_layerInfo[1].view(4, 65, 14, 14)  # 56
        v2_1 = self.vitLayer_UpConv(v2)
        v2_2 = self.vitLayer_UpConv(v2_1)

        v3 = vit_layerInfo[2].view(4, 65, 14, 14)  # 112
        v3_1 = self.vitLayer_UpConv(v3)
        v3_2 = self.vitLayer_UpConv(v3_1)
        v3_3 = self.vitLayer_UpConv(v3_2)

        v4 = vit_layerInfo[3].view(4, 65, 14, 14)  # 224
        v4_1 = self.vitLayer_UpConv(v4)
        v4_2 = self.vitLayer_UpConv(v4_1)
        v4_3 = self.vitLayer_UpConv(v4_2)
        v4_4 = self.vitLayer_UpConv(v4_3)

        xe1 = self.DoubleConv3_64(x)
        xe1_p = self.pool(xe1)
        xe2_p1 = torch.cat((v3_3, e1), dim=1)  # 65+64
        xe2_p2 = self.conv64(xe2_p1) + xe1_p
        xe2 = self.DoubleConv64_128(xe2_p2)
        xe2_p = self.pool(xe2)

        xe3_p1 = torch.cat((v2_2, e2), dim=1)  # 65+128
        xe3_p2 = self.conv128(xe3_p1) + xe2_p
        xe3 = self.DoubleConv128_256(xe3_p2)
        xe3_p = self.pool(xe3)

        xe4_p1 = torch.cat((v1_1, e3), dim=1)  # 65+256
        xe4_p2 = self.conv256(xe4_p1) + xe3_p
        xe4 = self.DoubleConv256_512(xe4_p2)
        xe4_p = self.pool(xe4)

        xe5_p1 = torch.cat((v1, e4), dim=1)  # 65+512
        xe5_p2 = self.conv512(xe5_p1) + xe4_p
        xe5 = self.DoubleConv512_1024(xe5_p2)

        xu1 = self.up128_64(xe2)

        xu2_1 = self.up256_128(xe3)
        xji_p = self.up128_128(xu2_1)
        xj1 = torch.cat((xu1, xji_p), dim=1)
        xj1 = self.conv64128_64(xj1)  # ([4, 64, 224, 224])

        xu2_2 = self.up128_64(xu2_1)

        xu3_1 = self.up512_256(xe4)
        xj2_p = self.up256_256(xu3_1)
        xj2 = torch.cat((xj2_p, xu2_1), dim=1)
        xj2 = self.conv128256_128(xj2)  # ([4, 128, 112, 112])

        xj3_p = self.up128_128(xj2)
        xj3 = torch.cat((xu2_2, xj3_p), dim=1)
        xj3 = self.conv64128_64(xj3)  # ([4, 64, 224, 224])

        xu3_2 = self.up256_128(xu3_1)
        xj4 = torch.cat((xu3_2, xj2), dim=1)
        xj4 = self.conv256_128(xj4)  # ([4, 128, 112, 112])

        xj5_p = self.up128_128(xj4)
        xj5 = torch.cat((xj3, xj5_p), dim=1)
        xj5 = self.conv64128_64(xj5)  # ([4, 64, 224, 224])

        xj6 = torch.cat((xj1, xu2_2), dim=1)
        xj6 = self.conv128_64(xj6)  # ([4, 64, 224, 224])

        xj7 = torch.cat((xj6, xj3), dim=1)
        xj7 = self.conv128_64(xj7)  # ([4, 64, 224, 224])

        xj8 = torch.cat((xj7, xj5), dim=1)
        xj8 = self.conv128_64(xj8)

        out = self.final_conv(xj8)
        return out


