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
from torch.distributions import Normal, Independent

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model

class SELayer(torch.nn.Module):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_double = torch.nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 16, 1, 1, 0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(num_filter // 16, num_filter, 1, 1, 0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(mask)
        x = x * mask
        return x


class ResBlock(nn.Module):
    def __init__(self, num_filter, time_emb_dim=None):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                body.append(nn.LeakyReLU())
        body.append(SELayer(num_filter))
        self.body = nn.Sequential(*body)

        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, num_filter)
        ) if exists(time_emb_dim) else None

    def forward(self, x, time_emb=None):
        res = self.body(x)
        if exists(self.mlp):
            res += self.mlp(time_emb)[:, :, None, None]
        x = res + x
        return x


class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, time_emb_dim=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(),
        )

        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, ch_out)
        ) if exists(time_emb_dim) else None

    def forward(self, x, time_emb):
        x = self.conv(x)
        if exists(self.mlp):
            x += self.mlp(time_emb)[:, :, None, None]
        return x

class Compute_z(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Compute_z, self).__init__()
        self.latent_dim = latent_dim
        self.u_conv_layer = nn.Conv2d(input_dim, 2 * self.latent_dim, kernel_size=1, padding=0)
        self.s_conv_layer = nn.Conv2d(input_dim, 2 * self.latent_dim, kernel_size=1, padding=0)

    def forward(self, x):
        u_encoding = torch.mean(x, dim=2, keepdim=True)
        u_encoding = torch.mean(u_encoding, dim=3, keepdim=True)
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


class Encoder(nn.Module):
    def __init__(self, ch, time_dim):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(ch_in=ch, ch_out=64, time_emb_dim=time_dim)
        self.conv2 = ConvBlock(ch_in=64, ch_out=64, time_emb_dim=time_dim)
        self.conv3 = ConvBlock(ch_in=64, ch_out=64, time_emb_dim=time_dim)
        self.conv4 = ResBlock(64, time_emb_dim=time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
    def forward(self, x, time):
        x1 = self.conv1(x, time)
        x2 = self.pool1(x1)
        x2 = self.conv2(x2, time)
        x3 = self.pool2(x2)
        x3 = self.conv3(x3, time)
        x4 = self.pool3(x3)
        x4 = self.conv4(x4, time)
        return x1, x2, x3, x4

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

        self.encoder_water = Encoder(in_channel, time_dim)
        self.encoder_air = Encoder(in_channel, time_dim)

        self.water_conv = ResBlock(64, time_emb_dim=time_dim)
        self.air_conv = ResBlock(64, time_emb_dim=time_dim)

        self.water_Up3 = Up()
        self.water_UpConv3 = ConvBlock(ch_in=128, ch_out=64, time_emb_dim=time_dim)
        self.water_Up2 = Up()
        self.water_UpConv2 = ConvBlock(ch_in=128, ch_out=64, time_emb_dim=time_dim)
        self.water_Up1 = Up()
        self.water_UpConv1 = ConvBlock(ch_in=128, ch_out=64, time_emb_dim=time_dim)

        self.air_Up3 = Up()
        self.air_UpConv3 = ConvBlock(ch_in=128, ch_out=64, time_emb_dim=time_dim)
        self.air_Up2 = Up()
        self.air_UpConv2 = ConvBlock(ch_in=128, ch_out=64, time_emb_dim=time_dim)
        self.air_Up1 = Up()
        self.air_UpConv1 = ConvBlock(ch_in=128, ch_out=64, time_emb_dim=time_dim)

        out_conv = []
        out_conv.append(ResBlock(64, time_emb_dim=None))
        out_conv.append(ResBlock(64, time_emb_dim=None))
        out_conv.append(nn.Conv2d(64, out_channel, kernel_size=1, padding=0))
        self.out_conv = nn.Sequential(*out_conv)

        z = 20

        self.compute_z_water = Compute_z(z, 128)
        self.compute_z_air = Compute_z(z, 128)

        self.conv_u = nn.Conv2d(z, 128, kernel_size=1, padding=0)
        self.conv_s = nn.Conv2d(z, 128, kernel_size=1, padding=0)

        self.IN = nn.InstanceNorm2d(128)

    def forward(self, x, x_air, time):
        # print(time.shape)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # print(x.shape)
        # print(t.shape)
        # feats = []
        x1, x2, x3, x4 = self.encoder_water(x, t)

        x1_air, x2_air, x3_air, x4_air = self.encoder_air(x_air, t)

        water_x4 = self.water_conv(x4, t)
        air_x4 = self.air_conv(x4_air, t)
        water_d3 = self.water_Up3(water_x4)
        air_d3 = self.air_Up3(air_x4)

        water_d3 = torch.cat((x3, water_d3), dim=1)
        air_d3 = torch.cat((x3_air, air_d3), dim=1)
        water_d3 = self.water_UpConv3(water_d3, t)
        air_d3 = self.air_UpConv3(air_d3, t)
        water_d2 = self.water_Up2(water_d3)
        air_d2 = self.air_Up2(air_d3)

        water_d2 = torch.cat((x2, water_d2), dim=1)
        air_d2 = torch.cat((x2_air, air_d2), dim=1)
        water_d2 = self.water_UpConv2(water_d2, t)
        air_d2 = self.air_UpConv2(air_d2, t)
        water_d1 = self.water_Up1(water_d2)
        air_d1 = self.air_Up1(air_d2)

        water_d1 = torch.cat((x1, water_d1), dim=1)
        air_d1 = torch.cat((x1_air, air_d1), dim=1)
        # x1->dis
        water_u_dist, water_s_dist, _, _, _, _ = self.compute_z_water(water_d1)
        air_u_dist, air_s_dist, _, _, _, _ = self.compute_z_air(air_d1)
        air_latent_u = air_u_dist.rsample()
        air_latent_s = air_s_dist.rsample()
        air_latent_u = torch.unsqueeze(air_latent_u, -1)
        air_latent_u = torch.unsqueeze(air_latent_u, -1)
        air_latent_s = torch.unsqueeze(air_latent_s, -1)
        air_latent_s = torch.unsqueeze(air_latent_s, -1)
        air_u = self.conv_u(air_latent_u)
        air_s = self.conv_s(air_latent_s)
        water_d1 = self.IN(water_d1) * torch.abs(air_s) + air_u
        # x1->out
        water_d1 = self.water_UpConv1(water_d1, t)
        out = self.out_conv(water_d1)
        air_d1 = self.air_UpConv1(air_d1, t)
        out_air = self.out_conv(air_d1)

        return out, out_air, \
               water_u_dist, water_s_dist, air_u_dist, air_s_dist

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

    img = torch.zeros(2, 3, 128, 128)
    time = torch.tensor([1, 2])
    model = UNet(inner_channel=32, in_channel=3)
    output, out2, a, b, c, d = model(img, img, time)
    # output = model2(img)
    print(output.shape)
    print(a)
    print(b)
    # total = sum([param.nelement() for param in model.parameters()])
    # print('parameter: %.2fM' % (total / 1e6))