import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from model.ddpm_trans_modules.trans_block_dual import TransformerBlock_dual
from model.ddpm_trans_modules.trans_block_eca import TransformerBlock_eca
from model.ddpm_trans_modules.trans_block_sa import TransformerBlock_sa
from model.ddpm_trans_modules.trans_block_sge import TransformerBlock_sge
import torchvision
from torch.distributions import Normal, Independent
from model.spatial_attention import SpatialTransformer


class AdaAttN(nn.Module):
    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        # mean = F.interpolate(mean, content.size()[2:])
        # std = F.interpolate(std, content.size()[2:])
        # print(mean.shape)
        # print(std.shape)
        # print(content.shape)
        return std * mean_variance_norm(content) + mean

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    # input = F.normalize(input, p=2, dim=1, eps=1e-12)
    # input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b)

def calc_mean_std(input, eps=1e-5):
    batch_size, channels = input.shape[:2]

    reshaped = input.view(batch_size, channels, -1) # Reshape channel wise
    mean = torch.mean(reshaped, dim = 2).view(batch_size, channels, 1, 1) # Calculat mean and reshape
    std = torch.sqrt(torch.var(reshaped, dim=2)+eps).view(batch_size, channels, 1, 1) # Calculate variance, add epsilon (avoid 0 division),
                                                                                # calculate std and reshape
    return mean, std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def AdaIn(content, style):
    assert content.shape[:2] == style.shape[:2] # Only first two dim, such that different image sizes is possible
    batch_size, n_channels = content.shape[:2]
    mean_content, std_content = calc_mean_std(content)
    mean_style, std_style = calc_mean_std(style)

    output = std_style*((content - mean_content) / (std_content)) + mean_style # Normalise, then modify mean and std
    return output

# model

class Compute_z(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Compute_z, self).__init__()
        self.latent_dim = latent_dim
        self.u_conv_layer = nn.Conv2d(input_dim, 2 * self.latent_dim, kernel_size=1, padding=0)
        self.s_conv_layer = nn.Conv2d(input_dim, 2 * self.latent_dim, kernel_size=1, padding=0)

    def forward(self, x):
        u_encoding = torch.mean(x, dim=2, keepdim=True)
        # print(u_encoding.shape)
        u_encoding = torch.mean(u_encoding, dim=3, keepdim=True)
        # print(u_encoding.shape)
        u_mu_log_sigma = self.u_conv_layer(u_encoding)
        u_mu_log_sigma = torch.squeeze(u_mu_log_sigma, dim=2)
        u_mu_log_sigma = torch.squeeze(u_mu_log_sigma, dim=2)
        u_mu = u_mu_log_sigma[:, :self.latent_dim]
        u_log_sigma = u_mu_log_sigma[:, self.latent_dim:]
        u_dist = Independent(Normal(loc=u_mu, scale=torch.exp(u_log_sigma)), 1)

        s_encoding = torch.std(x, dim=2, keepdim=True)
        s_encoding = torch.std(s_encoding, dim=3, keepdim=True)
        s_mu_log_sigma = self.s_conv_layer(s_encoding)
        s_mu_log_sigma = torch.squeeze(s_mu_log_sigma, dim=2)
        s_mu_log_sigma = torch.squeeze(s_mu_log_sigma, dim=2)
        s_mu = s_mu_log_sigma[:, :self.latent_dim]
        s_log_sigma = s_mu_log_sigma[:, self.latent_dim:]
        s_dist = Independent(Normal(loc=s_mu, scale=torch.exp(s_log_sigma)), 1)
        return u_dist, s_dist, u_mu, s_mu, torch.exp(u_log_sigma), torch.exp(s_log_sigma)

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

class Encoder(nn.Module):
    def __init__(
            self,
            in_channel=6,
            inner_channel=32,
            norm_groups=32,
    ):
        super().__init__()

        dim = inner_channel
        time_dim = inner_channel

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.PixelUnshuffle(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.block1 = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups,
                                     with_attn=True)
        self.block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block4 = ResnetBloc_eca(dim=dim * 2 ** 3, dim_out=dim * 2 ** 3, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)

        self.cross_att2 = SpatialTransformer(dim * 2 ** 1, 1, dim * 2 ** 1, context_dim=768)
        self.cross_att3 = SpatialTransformer(dim * 2 ** 2, 1, dim * 2 ** 2, context_dim=768)
        self.cross_att4 = SpatialTransformer(dim * 2 ** 3, 1, dim * 2 ** 3, context_dim=768)


        ####### control net #########
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            # nn.Conv2d(16, 16, 3, padding=1),
            # nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            # nn.Conv2d(32, 32, 3, padding=1),
            zero_module(nn.Conv2d(32, dim, 3, padding=1))
        )
        self.conv2_control = nn.Sequential(nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                                   nn.PixelUnshuffle(2))
        self.conv3_control = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 1) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))

        self.conv4_control = nn.Sequential(
            nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2) // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2))
        
        self.block1_control = ResnetBloc_eca(dim=dim, dim_out=dim, time_emb_dim=time_dim, norm_groups=norm_groups,
                                     with_attn=True)
        self.block2_control = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block3_control = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)
        self.block4_control = ResnetBloc_eca(dim=dim * 2 ** 3, dim_out=dim * 2 ** 3, time_emb_dim=time_dim,
                                     norm_groups=norm_groups, with_attn=True)

        # self.cross_att1_control = SpatialTransformer(dim, 1, dim, context_dim=768)
        # self.cross_att2_control = SpatialTransformer(dim * 2 ** 1, 1, dim * 2 ** 1, context_dim=768)
        # self.cross_att3_control = SpatialTransformer(dim * 2 ** 2, 1, dim * 2 ** 2, context_dim=768)
        # self.cross_att4_control = SpatialTransformer(dim * 2 ** 3, 1, dim * 2 ** 3, context_dim=768)

        # self.cross_att12_control = SpatialTransformer(dim, 1, dim)
        # self.cross_att22_control = SpatialTransformer(dim * 2 ** 1, 1, dim * 2 ** 1)
        # self.cross_att32_control = SpatialTransformer(dim * 2 ** 2, 1, dim * 2 ** 2)
        # self.cross_att42_control = SpatialTransformer(dim * 2 ** 3, 1, dim * 2 ** 3)

        # self.block1_zero_control = nn.Sequential(zero_module(nn.Conv2d(dim, dim, 3, padding=1)))
        # self.block2_zero_control = nn.Sequential(zero_module(nn.Conv2d(dim * 2 ** 1, dim * 2 ** 1, 3, padding=1)))
        # self.block3_zero_control = nn.Sequential(zero_module(nn.Conv2d(dim * 2 ** 2, dim * 2 ** 2, 3, padding=1)))
        # self.block4_zero_control = nn.Sequential(zero_module(nn.Conv2d(dim * 2 ** 3, dim * 2 ** 3, 3, padding=1)))

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

        self.decoder_block3 = ResnetBloc_eca(dim=dim * 2 ** 2, dim_out=dim * 2 ** 2, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block2 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)
        self.decoder_block1 = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
                                             norm_groups=norm_groups, with_attn=True)

        # self.cross_att1_de = SpatialTransformer(dim * 2 ** 1, 1, dim * 2 ** 1, context_dim=768)
        self.cross_att2_de = SpatialTransformer(dim * 2 ** 1, 1, dim * 2 ** 1, context_dim=768)
        self.cross_att3_de = SpatialTransformer(dim * 2 ** 2, 1, dim * 2 ** 2, context_dim=768)

        ########## style transfer ##########
        # self.compute_z_water = Compute_z(32, input_dim=dim * 2 ** 1)
        # self.compute_z_air = Compute_z(32, input_dim=dim * 2 ** 1)
        # self.conv_u = nn.Conv2d(32, dim * 2 ** 1, kernel_size=1, padding=0)
        # self.conv_s = nn.Conv2d(32, dim * 2 ** 1, kernel_size=1, padding=0)
        # self.AdaIN = nn.InstanceNorm2d(dim * 2 ** 1)
        self.adaattn1 = AdaAttN(dim, 128 * 128, dim + (dim * 2 ** 1) + (dim * 2 ** 2) + (dim * 2 ** 3))
        self.adaattn2 = AdaAttN(dim * 2 ** 1, 128 * 128, dim + (dim * 2 ** 1) + (dim * 2 ** 2) + (dim * 2 ** 3))
        self.adaattn3 = AdaAttN(dim * 2 ** 2, 128 * 128, dim + (dim * 2 ** 1) + (dim * 2 ** 2) + (dim * 2 ** 3))
        self.adaattn4 = AdaAttN(dim * 2 ** 3, 128 * 128, dim + (dim * 2 ** 1) + (dim * 2 ** 2) + (dim * 2 ** 3))
    def forward(self, x, t, style, context):
        x_style = self.input_hint_block(style)
        x = self.conv1(x)
        x1 = self.block1(x, t)

        x2 = self.conv2(x1)
        x2 = self.block2(x2, t)
        x2 = self.cross_att2(x2, context)

        x3 = self.conv3(x2)
        x3 = self.block3(x3, t)
        x3 = self.cross_att3(x3, context)

        x4 = self.conv4(x3)
        x4 = self.block4(x4, t)
        x4 = self.cross_att4(x4, context)

        ######## control net forward ##########
        # x_style = AdaIn(x, x_style)
        # x_context = x_style + x
        x_context1 = self.block1_control(x_style, t)
        # x_context1 = self.cross_att1_control(x_context1, context)
        # x_context1 = self.cross_att12_control(x_context1, x1)
        # print('context1 shape:', x_context1.shape)
        x_context2 = self.conv2_control(x_context1)
        x_context2 = self.block2_control(x_context2, t)
        # x_context2 = self.cross_att2_control(x_context2, context)
        # x_context2 = self.cross_att22_control(x_context2, x2)

        x_context3 = self.conv3_control(x_context2)
        x_context3 = self.block3_control(x_context3, t)
        # x_context3 = self.cross_att3_control(x_context3, context)
        # x_context3 = self.cross_att32_control(x_context3, x3)

        x_context4 = self.conv4_control(x_context3)
        x_context4 = self.block4_control(x_context4, t)
        # x_context4 = self.cross_att4_control(x_context4, context)
        # x_context4 = self.cross_att42_control(x_context4, x4)

        # x_context1_de = self.block1_zero_control(x_context1)
        # x_context2_de = self.block2_zero_control(x_context2)
        # x_context3_de = self.block3_zero_control(x_context3)
        # x_context4_de = self.block4_zero_control(x_context4)

        ######### style AdaAttN ###########
        feat_scales = torch.cat([x1, F.interpolate(x2, (x1.size()[2:])),
                                 F.interpolate(x3, (x1.size()[2:])), F.interpolate(x4, (x1.size()[2:]))], dim=1)
        feat_scales_context = torch.cat([x_context1, F.interpolate(x_context2, (x_context1.size()[2:])),
                                         F.interpolate(x_context3, (x_context1.size()[2:])),
                                         F.interpolate(x_context4, (x_context1.size()[2:]))], dim=1)

        feat_scales2 = torch.cat([F.interpolate(x1, (x2.size()[2:])), x2,
                                  F.interpolate(x3, (x2.size()[2:])), F.interpolate(x4, (x2.size()[2:]))], dim=1)
        feat_scales_context2 = torch.cat([F.interpolate(x_context1, (x_context2.size()[2:])), x_context2,
                                          F.interpolate(x_context3, (x_context2.size()[2:])),
                                          F.interpolate(x_context4, (x_context2.size()[2:]))], dim=1)

        feat_scales3 = torch.cat([F.interpolate(x1, (x3.size()[2:])), F.interpolate(x2, (x3.size()[2:])),
                                  x3, F.interpolate(x4, (x3.size()[2:]))], dim=1)
        feat_scales_context3 = torch.cat(
            [F.interpolate(x_context1, (x_context3.size()[2:])), F.interpolate(x_context2, (x_context3.size()[2:])),
             x_context3,
             F.interpolate(x_context4, (x_context3.size()[2:]))], dim=1)

        feat_scales4 = torch.cat([F.interpolate(x1, (x4.size()[2:])), F.interpolate(x2, (x4.size()[2:])),
                                  F.interpolate(x3, (x4.size()[2:])), x4], dim=1)
        feat_scales_context4 = torch.cat(
            [F.interpolate(x_context1, (x_context4.size()[2:])), F.interpolate(x_context2, (x_context4.size()[2:])),
             F.interpolate(x_context3, (x_context4.size()[2:])),
             x_context4], dim=1)
        ######## decoder forward ##########
        # x4 = AdaIn(x4, x_context4)
        x4 = self.adaattn4(x4, x_context4, feat_scales4, feat_scales_context4)
        # x4 = x4 + x_context4_de
        de_level3 = self.conv_up3(x4)
        de_level3 = torch.cat([de_level3, x3], 1)
        de_level3 = self.conv_cat3(de_level3)

        # de_level3 = AdaIn(de_level3, x_context3)
        de_level3 = self.adaattn3(de_level3, x_context3, feat_scales3, feat_scales_context3)
        # de_level3 = de_level3 + x_context3_de
        de_level3 = self.decoder_block3(de_level3, t)
        de_level3 = self.cross_att3_de(de_level3, context)
        de_level2 = self.conv_up2(de_level3)
        de_level2 = torch.cat([de_level2, x2], 1)
        de_level2 = self.conv_cat2(de_level2)

        # de_level2 = AdaIn(de_level2, x_context2)
        de_level2 = self.adaattn2(de_level2, x_context2, feat_scales2, feat_scales_context2)
        # de_level2 = de_level2 + x_context2_de
        de_level2 = self.decoder_block2(de_level2, t)
        de_level2 = self.cross_att2_de(de_level2, context)
        de_level1 = self.conv_up1(de_level2)

        # de_level1 = AdaIn(de_level1, x_context1)
        de_level2 = self.adaattn1(de_level1, x_context1, feat_scales, feat_scales_context)
        # de_level1 = de_level1 + x_context1_de
        de_level1 = torch.cat([de_level1, x1], 1)

        mid_feat = self.decoder_block1(de_level1, t)
        # mid_feat = AdaIn(mid_feat, x_context2)
        mid_feat = self.adaattn2(mid_feat, x_context2,
                                 F.interpolate(feat_scales2, context.size()[2:]),
                                 F.interpolate(feat_scales_context2, context.size()[2:]))
        # print(mid_feat.shape)
        # print(x_context2_de.shape)
        # mid_feat = AdaIn(mid_feat, x_style)
        # mid_feat = self.cross_att1_de(mid_feat, context)
        '''
        ######## control decoder forward ##########
        # x4 = AdaIn(x4, x_context4)
        # x4 = x4 + x_context4_de
        de_level3_context = self.conv_up3(x_context4)
        de_level3_context = torch.cat([de_level3_context, x_context3], 1)
        de_level3_context = self.conv_cat3(de_level3_context)

        # de_level3 = AdaIn(de_level3, x_context3)
        # de_level3 = de_level3 + x_context3_de
        de_level3_context = self.decoder_block3(de_level3_context, t)
        # de_level3 = self.cross_att3_de(de_level3, context)
        de_level2_context = self.conv_up2(de_level3_context)
        de_level2_context = torch.cat([de_level2_context, x_context2], 1)
        de_level2_context = self.conv_cat2(de_level2_context)

        # de_level2 = AdaIn(de_level2, x_context2)
        # de_level2 = de_level2 + x_context2_de
        de_level2_context = self.decoder_block2(de_level2_context, t)
        # de_level2 = self.cross_att2_de(de_level2, context)
        de_level1_context = self.conv_up1(de_level2_context)

        # de_level1 = AdaIn(de_level1, x_context1)
        # de_level1 = de_level1 + x_context1_de
        de_level1_context = torch.cat([de_level1_context, x_context1], 1)

        mid_feat_context = self.decoder_block1(de_level1_context, t)
        
        ########## style transfer ##########
        # water_u_dist, water_s_dist, _, _, _, _ = self.compute_z_water(mid_feat)
        # air_u_dist, air_s_dist, _, _, _, _ = self.compute_z_air(mid_feat_context)
        # air_latent_u = air_u_dist.rsample()
        # air_latent_s = air_s_dist.rsample()
        # air_latent_u = torch.unsqueeze(air_latent_u, -1)
        # air_latent_u = torch.unsqueeze(air_latent_u, -1)
        # air_latent_s = torch.unsqueeze(air_latent_s, -1)
        # air_latent_s = torch.unsqueeze(air_latent_s, -1)
        # air_u = self.conv_u(air_latent_u)
        # air_s = self.conv_s(air_latent_s)
        #
        # mid_feat = self.AdaIN(mid_feat) * torch.abs(air_s) + air_u
        '''
        return mid_feat, de_level2

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = 1e-5

    def forward(self, x, y):
        mean_x, mean_y = torch.mean(x, dim=(2, 3), keepdim=True), torch.mean(y, dim=(2, 3), keepdim=True)
        std_x, std_y = torch.std(x, dim=(2, 3), keepdim=True) + self.eps, torch.std(y, dim=(2, 3), keepdim=True) + self.eps
        return std_y * (x - mean_x) / std_x + mean_y

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

        self.encoder_water = Encoder(in_channel=in_channel, inner_channel=inner_channel, norm_groups=norm_groups)
        # self.encoder_air = Encoder(in_channel=in_channel, inner_channel=inner_channel, norm_groups=norm_groups)

        self.refine = ResnetBloc_eca(dim=dim*2**1, dim_out=dim*2**1, time_emb_dim=time_dim, norm_groups=norm_groups, with_attn=True)
        self.de_predict = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))
        self.de_predict_support = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))
        # self.de_predict_support2 = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))
        # self.upsample = torch.nn.functional.interpolate
        # self.refine_air = ResnetBloc_eca(dim=dim * 2 ** 1, dim_out=dim * 2 ** 1, time_emb_dim=time_dim,
        #                              norm_groups=norm_groups, with_attn=True)
        # self.de_predict_air = nn.Sequential(nn.Conv2d(dim * 2 ** 1, out_channel, kernel_size=1, stride=1))

        # self.compute_z_water = Compute_z(32, input_dim=dim * 2 ** 1)
        # self.compute_z_air = Compute_z(32, input_dim=dim * 2 ** 1)
        # self.conv_u = nn.Conv2d(32, dim * 2 ** 1, kernel_size=1, padding=0)
        # self.conv_s = nn.Conv2d(32, dim * 2 ** 1, kernel_size=1, padding=0)
        # self.AdaIN = AdaptiveInstanceNorm2d()

    def forward(self, x, time, style, context):
        # print(time.shape)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # print(x.shape)
        # print(t.shape)
        # feats = []
        mid_feat, x1 = self.encoder_water(x, t, style, context)
        # mid_feat_air, x1_air = self.encoder_air(x_air, t)

        mid_feat2 = self.refine(mid_feat, t)
        # print(mid_feat2.shape)
        # mid_feat_air = self.refine_air(mid_feat_air, t)
        # x1 = self.upsample(x1, mode='bilinear', size=mid_feat2.size()[2:], align_corners=False)
        ####### exchange style code ############
        # mid_feat_exchange = self.AdaIN(mid_feat, mid_feat_air)
        # water_u_dist, water_s_dist, _, _, _, _ = self.compute_z_water(mid_feat)
        # air_u_dist, air_s_dist, _, _, _, _ = self.compute_z_air(mid_feat_air)
        # air_latent_u = air_u_dist.rsample()
        # air_latent_s = air_s_dist.rsample()
        # air_latent_u = torch.unsqueeze(air_latent_u, -1)
        # air_latent_u = torch.unsqueeze(air_latent_u, -1)
        # air_latent_s = torch.unsqueeze(air_latent_s, -1)
        # air_latent_s = torch.unsqueeze(air_latent_s, -1)
        # air_u = self.conv_u(air_latent_u)
        # air_s = self.conv_s(air_latent_s)
        #
        # mid_feat = self.IN(mid_feat) * torch.abs(air_s) + air_u
        # print(self.IN(mid_feat).shape)
        # print(mid_feat.shape)
        # print(air_s.shape)
        # print(air_u.shape)
        # mid_feat_air = self.IN(mid_feat_air) * torch.abs(air_s) + air_u

        # return self.de_predict(mid_feat2), self.de_predict_support(x1), self.de_predict_support(mid_feat)
        return self.de_predict(mid_feat2)
        # return self.de_predict(mid_feat), self.de_predict_air(mid_feat_air), gram_matrix(mid_feat), gram_matrix(mid_feat_air)

if __name__ == '__main__':
    # img = torch.zeros(4, 128, 128, 128)
    # compute_z = Compute_z(20)
    # conv_u = nn.Conv2d(20, 128, kernel_size=1, padding=0)
    # insnorm = nn.InstanceNorm2d(128)
    # a, b, _, _, _, _ = compute_z(img)
    # print(a.rsample().shape)
    # po_latent_u = torch.unsqueeze(a.rsample(), -1)
    # po_latent_u = torch.unsqueeze(po_latent_u, -1)
    # c = conv_u(po_latent_u)
    # print(c.shape)
    # d = insnorm(img) + c
    # print(d.shape)
    # a = AdaAttN(96, 128 * 128)
    #
    # a_input = torch.zeros(2, 96, 128, 128)
    # b_input = torch.zeros(2, 96, 128, 128)
    #
    # c_input = torch.zeros(2, 96 * 2, 128, 128)
    # d_input = torch.zeros(2, 96 * 2, 128, 128)
    #
    # a_output = a(a_input, b_input, c_input, d_input)

    from model.utils import load_part_of_model2
    param_path = '/home/ty/code/DM_underwater/experiments_train/sr_ffhq_230922_155247/checkpoint/I200000_E710_gen.pth'
    img = torch.zeros(2, 4, 128, 128)
    time = torch.tensor([1, 2])
    style = torch.zeros(2, 3, 128, 128)
    context = torch.zeros(2, 77, 768)
    model = UNet(inner_channel=48, norm_groups=24, in_channel=4)

    # model = load_part_of_model2(model, param_path)
    output = model(img, time, style, context)
    output = output.data.cpu().numpy()
    # output = model2(img)
    print(output.shape)
    # from matplotlib import pyplot as plt
    # plt.imshow(output[0, 0, :, :])
    # plt.show()
    # print(a.shape)
    # print(b)
    # total = sum([param.nelement() for param in model.parameters()])
    # print('parameter: %.2fM' % (total / 1e6))

