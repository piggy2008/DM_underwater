from model.ddpm_trans_modules.unet_original import UNet
from model.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from model.utils import load_part_of_model2
from PIL import Image
import torchvision
import torch
import os
from matplotlib import pyplot as plt
import numpy as np
from functools import partial

to_torch = partial(torch.tensor, dtype=torch.float32, device=0)

def conver2image(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

device = 0
src_model_path = 'experiments_supervised/I1000000_E1777_gen.pth'
betas = np.linspace(1e-6, 1e-2, 1000, dtype=np.float64)
betas = to_torch(betas)
# noise_schedule = NoiseScheduleVP(schedule='linear', continuous_beta_0=1e-6, continuous_beta_1=1e-2)
# noise_schedule = NoiseScheduleVP(schedule='linear', continuous_beta_0=0.1, continuous_beta_1=20.)
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)
model = UNet(inner_channel=48, norm_groups=24, in_channel=6).to(device=device)

model = load_part_of_model2(model, src_model_path)
# print(model)
totensor = torchvision.transforms.ToTensor()

image_root = 'dataset/water_val_16_128'
image_name = '00000.png'

image_names = os.listdir(os.path.join(image_root, 'sr_16_128'))

for name in image_names:
    sr = Image.open(os.path.join(image_root, 'sr_16_128', name)).convert("RGB")
    hr = Image.open(os.path.join(image_root, 'hr_128', name)).convert("RGB")
    sr = totensor(sr) * 2 - 1
    hr = totensor(hr) * 2 - 1
    sr = torch.unsqueeze(sr, 0).to(device=device)
    hr = torch.unsqueeze(hr, 0).to(device=device)
    x_T = torch.randn(sr.shape).to(device=device)

    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="noise",  # or "x_start" or "v" or "score"
        guidance_type="classifier-free",
        condition=sr,
        unconditional_condition=None,
        guidance_scale=1,
    )
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
    x_sample = dpm_solver.sample(
        x_T,
        steps=10,
        order=1,
        skip_type="time_uniform",
        method="multistep",
    )
    print(x_sample.shape)
    # print(x_sample)
    img = conver2image(x_sample)
    img = Image.fromarray(img, "RGB")
    img.save(os.path.join('results', name))
# print(np.unique(x_sample))
# img = conver2image(x_sample)
#
# plt.subplot(1, 3, 1)
# plt.imshow(conver2image(sr))
#
# plt.subplot(1, 3, 2)
# plt.imshow(conver2image(hr))
#
# plt.subplot(1, 3, 3)
# plt.imshow(img)
# plt.show()

