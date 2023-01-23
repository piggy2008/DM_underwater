import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from model.ddpm_trans_modules.trans_block_normal import TransformerBlock
from model.ddpm_trans_modules.trans_block_dual import TransformerBlock_dual
from model.ddpm_trans_modules.trans_block_eca import TransformerBlock_eca
from model.ddpm_trans_modules.trans_block_sa import TransformerBlock_sa
from model.ddpm_trans_modules.trans_block_sge import TransformerBlock_sge
import torchvision

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class ResnetBloc_da(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = nn.Sequential(*[TransformerBlock_dual(dim=int(dim), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class ResnetBloc_eca(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = nn.Sequential(*[TransformerBlock_eca(dim=int(dim), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class ResnetBloc_norm(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class ResnetBloc_sge(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = nn.Sequential(*[TransformerBlock_sge(dim=int(dim), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class ResnetBloc_sa(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = nn.Sequential(*[TransformerBlock_sa(dim=int(dim), num_heads=2, ffn_expansion_factor=2.66,
                               bias=False, LayerNorm_type='WithBias') for i in range(1)])

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        dim = inner_channel

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                                       nn.PixelUnshuffle(2))
        self.conv3 = nn.Sequential(nn.Conv2d(int(dim*2**1), int(dim*2**1) // 2, kernel_size=3, stride=1, padding=1),
                                       nn.PixelUnshuffle(2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.block1 = ResnetBlocWithAttn(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=False)
        self.block2 = ResnetBloc_da(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)
        self.block3 = ResnetBloc_da(dim=dim*2**2, dim_out=dim*2**2, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)
        self.block4 = ResnetBloc_norm(dim=dim*2**3, dim_out=dim*2**3, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)

        self.conv_up3 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 3), (dim * 2 ** 3) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_up2 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 2), (dim * 2 ** 2) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))
        self.conv_up1 = nn.Sequential(
            nn.Conv2d((dim * 2 ** 1), (dim * 2 ** 1) * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv_cat3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=False)
        self.conv_cat2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=False)

        self.decoder_block3 = ResnetBlocWithAttn(dim=dim*2**2, dim_out=dim*2**2, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=False)
        self.decoder_block2 = ResnetBloc_da(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)
        self.decoder_block1 = ResnetBloc_da(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)

        self.refine = ResnetBloc_norm(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)
        self.de_predict = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))

        self.style_extractor = StyleFeatures(dim=dim)
        # self.fuse_feat1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        # self.fuse_feat2 = nn.Conv2d(int(dim*2**1) * 2, int(dim*2**1), kernel_size=1, bias=False)
        # self.fuse_feat3 = nn.Conv2d(int(dim*2**2) * 2, int(dim*2**2), kernel_size=1, bias=False)
        # self.fuse_feat4 = nn.Conv2d(int(dim*2**3) * 2, int(dim*2**3), kernel_size=1, bias=False)

    def forward(self, x, time, style_x=None):
        # print(time.shape)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # print(t.shape)
        # feats = []

        x = self.conv1(x)
        x1 = self.block1(x, t)

        x2 = self.conv2(x1)
        x2 = self.block2(x2, t)

        x3 = self.conv3(x2)
        x3 = self.block3(x3, t)

        x4 = self.conv4(x3)
        x4 = self.block4(x4, t)

        # if style_x is not None:
        #     feat1, feat2, feat3, feat4 = self.style_extractor(style_x)
        #     x1 = self.fuse_feat1(torch.cat([x1, feat1], dim=1))
        #     x2 = self.fuse_feat2(torch.cat([x2, feat2], dim=1))
        #     x3 = self.fuse_feat3(torch.cat([x3, feat3], dim=1))
        #     x4 = self.fuse_feat4(torch.cat([x4, feat4], dim=1))

        de_level3 = self.conv_up3(x4)
        de_level3 = torch.cat([de_level3, x3], 1)
        de_level3 = self.conv_cat3(de_level3)
        de_level3 = self.decoder_block3(de_level3, t)

        de_level2 = self.conv_up2(de_level3)
        de_level2 = torch.cat([de_level2, x2], 1)
        de_level2 = self.conv_cat2(de_level2)
        de_level2 = self.decoder_block2(de_level2, t)

        de_level1 = self.conv_up1(de_level2)
        de_level1 = torch.cat([de_level1, x1], 1)
        mid_feat = self.decoder_block1(de_level1, t)
        mid_feat = self.refine(mid_feat, t)

        return self.de_predict(mid_feat)

class StyleFeatures(torch.nn.Module):
    def __init__(self, dim=48):
        super(StyleFeatures, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.feat1_encode = nn.Sequential(nn.Conv2d(64, dim, kernel_size=3, stride=1, padding=1),
                                          TransformerBlock_sa(dim=int(dim), num_heads=2, ffn_expansion_factor=2.66,
                                                                bias=False, LayerNorm_type='WithBias'))
        self.feat2_encode = nn.Sequential(nn.Conv2d(128, dim*2**1, kernel_size=3, stride=1, padding=1),
                                          TransformerBlock_sa(dim=int(dim*2**1), num_heads=2, ffn_expansion_factor=2.66,
                                                                bias=False, LayerNorm_type='WithBias'))
        self.feat3_encode = nn.Sequential(nn.Conv2d(256, dim*2**2, kernel_size=3, stride=1, padding=1),
                                          TransformerBlock_sa(dim=int(dim*2**2), num_heads=2, ffn_expansion_factor=2.66,
                                                              bias=False, LayerNorm_type='WithBias'))
        self.feat4_encode = nn.Sequential(nn.Conv2d(512, dim*2**3, kernel_size=3, stride=1, padding=1),
                                          TransformerBlock_sa(dim=int(dim*2**3), num_heads=2, ffn_expansion_factor=2.66,
                                                              bias=False, LayerNorm_type='WithBias'))

    def forward(self, input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        input = (input + 1) / 2
        input = (input - self.mean) / self.std

        feat1 = self.blocks[0](input)
        feat2 = self.blocks[1](feat1)
        feat3 = self.blocks[2](feat2)
        feat4 = self.blocks[3](feat3)

        feat1 = self.feat1_encode(feat1)
        feat2 = self.feat2_encode(feat2)
        feat3 = self.feat3_encode(feat3)
        feat4 = self.feat4_encode(feat4)

        return feat1, feat2, feat3, feat4

if __name__ == '__main__':
    img = torch.zeros(4, 6, 128, 128)
    style_img = torch.zeros(4, 3, 128, 128)
    time = torch.tensor([1, 2, 3, 4])
    model = UNet(inner_channel=48, norm_groups=24)
    # model2 = StyleFeatures()
    output = model(img, time, style_img)
    # output = model2(img)
    print(output.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print('parameter: %.2fM' % (total / 1e6))