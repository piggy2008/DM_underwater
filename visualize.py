from PIL import Image
import numpy as np
import torchvision
import torch
from model.ddpm_trans_modules.unet import UNet
from model.utils import load_part_of_model2
import nethook
import os
from matplotlib import pyplot as plt
import core.metrics as Metrics
from tqdm import tqdm


betas = np.linspace(1e-6, 1e-2, 1000, dtype=np.float64)
# betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
alphas = 1. - betas
alphas_cumprod = np.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod))
sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod))
eta = 0
denoise_fn = None

def p_sample_ddim2(x, t, t_next, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
    b, *_, device = *x.shape, x.device
    bt = extract(betas, t, x.shape)
    at = extract((1.0 - betas).cumprod(), t, x.shape)
    print('at=', at)
    if condition_x is not None:
        et = denoise_fn(torch.cat([condition_x, x], dim=1), t)
    else:
        et = denoise_fn(x, t)

    x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
    # x0_air_t = (x_air - et_air * (1 - at).sqrt()) / at.sqrt()
    if t_next == None:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - betas).cumprod(), t_next, x.shape)
    if eta == 0:
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        # xt_air_next = at_next.sqrt() * x0_air_t + (1 - at_next).sqrt() * et_air
    elif at > (at_next):
        print('Inversion process is only possible with eta = 0')
        raise ValueError
    else:
        c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x0_t)
        # xt_air_next = at_next.sqrt() * x0_air_t + c2 * et_air + c1 * torch.randn_like(x0_t)

    # noise = noise_like(x.shape, device, repeat_noise)
    # no noise when t == 0

    return xt_next

def p_sample_loop2(sr, continous=False):
    sample_inter = 10
    x = sr
    condition_x = torch.mean(x, dim=1, keepdim=True)
    shape = x.shape
    b = shape[0]
    img = torch.randn(shape, device=device)
    start = 1000

    # t = torch.full((b,), start, device=device, dtype=torch.long)

    # noise = torch.randn(shape, device=device)
    # img = self.q_sample(x_start=x, t=t, noise=noise)
    # img_air = q_sample(x_start=x_air, t=t, noise=noise)

    ret_img = img
    # num_timesteps_ddim = np.array([0, 245, 521, 1052, 1143, 1286, 1475, 1587, 1765, 1859])  # searching
    c = 100
    num_timesteps_ddim = np.asarray(list(range(0, start, c)))
    time_steps = np.flip(num_timesteps_ddim[:-1])
    for j, i in enumerate(tqdm(time_steps, desc='sampling loop time step', total=len(time_steps))):
        # print('i = ', i)
        t = torch.full((b,), i, device=device, dtype=torch.long)
        if j == len(time_steps) - 1:
            t_next = None
        else:
            t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
        img = p_sample_ddim2(img, t, t_next, condition_x=condition_x)
        # print('i=', i)
        if i % sample_inter == 0:
            ret_img = torch.cat([ret_img, img], dim=0)

    if continous:
        return ret_img
    else:
        return ret_img[-1]

def conver2image(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def q_sample(x_start, t, noise=None):
    # noise = default(noise, lambda: torch.randn_like(x_start))
    if noise is None:
        noise = torch.randn_like(x_start)
    # fix gama
    return (
        extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        extract(sqrt_one_minus_alphas_cumprod,
                t, x_start.shape) * noise
    )

def q_sample_recover(x_noisy, t, predict_noise=None):
    # noise = default(noise, lambda: torch.randn_like(x_start))
    return (x_noisy - extract(sqrt_one_minus_alphas_cumprod,
                t, x_noisy.shape) * predict_noise) / extract(sqrt_alphas_cumprod, t, x_noisy.shape)

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out

src_model_path = 'experiments_train/sr_ffhq_230724_095846/checkpoint/I250000_E887_gen.pth'
image_root = 'dataset/water_val_16_128'
image_name = '00001.png'
device = 0
check_layer = 'encoder_water.block3'
# units = [0, 1, 2, 3, 4, 5, 6]
# units = [28, 20] # conv4
# units = [38, 20, 16, 21, 35, 9, 34, 1, 0, 3, 2, 8] # conv3
# units = [20, 23, 18, 22] # conv2
units = [8] # conv1
# units = range(0, 48)

totensor = torchvision.transforms.ToTensor()

sr = Image.open(os.path.join(image_root, 'sr_16_128', image_name)).convert("RGB")
hr = Image.open(os.path.join(image_root, 'hr_128', image_name)).convert("RGB")
sr = totensor(sr) * 2 - 1
hr = totensor(hr) * 2 - 1
sr = torch.unsqueeze(sr, 0)
hr = torch.unsqueeze(hr, 0)
# t = torch.full((1,), 100, dtype=torch.long)
# sr = q_sample(sr, t)
# hr = q_sample(hr, t)

sr = sr.to(device=device)
hr = hr.to(device=device)
# t = t.to(device=device)

model = UNet(inner_channel=48, norm_groups=24, in_channel=4).to(device=device)

model = load_part_of_model2(model, src_model_path)
print(model)

if not isinstance(model, nethook.InstrumentedModel):
    model = nethook.InstrumentedModel(model)
# model.retain_layer(check_layer)
# img = model(sr, hr, t)
# acts = model.retained_layer(check_layer)

# print(acts.shape)


#
set_units = units
def zero_out_tree_units(data, model):
    data[:, set_units, :, :] = 0.0
    return data

def turn_off_tree_units(data, model):
    data[:, set_units, :, :] = -5.0
    return data

# def update_tree_units(data, model):
#     mean, std = torch.std_mean(data[:, units, :, :], dim=[1, 2], keepdim=True)
#     mean_new, std_new = torch.std_mean(acts[:, units, :, :], dim=[1, 2], keepdim=True)
#     data[:, units, :, :] = std_new * (data[:, units, :, :] - mean) / std + mean_new
#     return data

# model.edit_layer('encoder_water.conv1', rule=turn_off_tree_units)
# img, hr_recover, _, _, _, _ = model(sr, hr, t)

# image2show = conver2image(img)
# gt = conver2image(sr)
# psnr = Metrics.calculate_psnr(image2show, gt)
# print(psnr)
#
# model(img)
# ranking = []
# for num in units:
#     set_units.append(num)
#     model.edit_layer(check_layer, rule=turn_off_tree_units)
#     img, _, _, _ = model(sr, hr, t)
#     x_0_recover = q_sample_recover(sr, t, predict_noise=img)
# # print(img.shape)
#     image2show = conver2image(x_0_recover)
#     gt = conver2image(sr)
#     psnr = Metrics.calculate_psnr(image2show, gt)
#     ranking.append([num, psnr])
#     set_units.clear()
# ranking.sort(key=lambda x: x[1])
# print (*ranking, sep="\n")

######## final output ##########
model.edit_layer(check_layer, rule=turn_off_tree_units)
denoise_fn = model
r = p_sample_loop2(sr, continous=False)

######## side output ##########
# condition_x = torch.mean(sr, dim=1, keepdim=True)
# model.retain_layer(check_layer)
# condition_x = torch.mean(hr, dim=1, keepdim=True)
#
# img = model(torch.cat([condition_x, sr], dim=1), hr, t)
# r = model.retained_layer(check_layer)
# r = r.data.cpu().numpy()

# img = model(sr, hr, t)
# img = model(torch.cat([condition_x, sr], dim=1), hr, t)
# x_0_recover = q_sample_recover(sr, t, predict_noise=img)
# at = extract((1.0 - betas).cumprod(), t, img.shape)
# x0_t = (sr - img * (1 - at).sqrt()) / at.sqrt()
# at_next = torch.ones_like(at)
# xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * img

plt.subplot(1, 3, 1)
plt.imshow(conver2image(sr))

plt.subplot(1, 3, 2)
plt.imshow(conver2image(hr))

plt.subplot(1, 3, 3)
plt.imshow(conver2image(r))
plt.show()



