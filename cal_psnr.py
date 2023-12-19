import os
import numpy as np
from PIL import Image
import math
import cv2
from SSIM_PIL import compare_ssim

from matplotlib import pyplot as plt
# from pyecharts import options as opts
# from pyecharts.charts import Bar

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


ckpt_path = '/home/ty/data/underwater/LSUI/HLRP'
gt_path = '/home/ty/data/underwater/LSUI/gt'

psnr_record = AvgMeter()
ssim_record = AvgMeter()
results = {}

images = os.listdir(gt_path)
images.sort()
image_names = []
psnr_list = []
for name in images:
    name_, suffix = os.path.splitext(name)
    print(name_)
    img = Image.open(os.path.join(ckpt_path, name_ + '.jpg')).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)

    gt = Image.open(os.path.join(gt_path, name)).convert('RGB')
    gt = gt.resize((256, 256))
    gt_array = np.array(gt)
    psnr = calculate_psnr(img_array, gt_array)
    ssim_ = compare_ssim(img, gt)
    psnr_record.update(psnr)
    ssim_record.update(ssim_)
    # each = {'name': name, 'psnr': psnr}
    image_names.append(name)
    psnr_list.append(psnr)

results['LSUI'] = {'PSNR': psnr_record.avg, 'SSIM': ssim_record.avg}
print(results)

# Twin adversarial contrastive learning for underwater image enhancement and beyond (TIP 2022)
# {'UIEB': {'PSNR': 23.093333269208358, 'SSIM': 0.8838852187072839}}
# {'LSUI': {'PSNR': 20.696740431218338, 'SSIM': 0.8228309119974603}}

# Underwater image enhancement with hyper-laplacian reflectance priors (TIP 2022)
# {'UIEB': {'PSNR': 12.56425772699218, 'SSIM': 0.2514555993612376}}
# {'LSUI': {'PSNR': 12.640982129405577, 'SSIM': 0.19295197254514615}}

# DiV LSUI 32.91M
# Validation # PSNR: 2.5946e+01 SSIM: 9.1482e-01
# DiV UIEB
# Validation # PSNR: 2.3055e+01 SSIM: 9.0094e-01

# UNET LSUI 32.23Mx
# Validation # PSNR: 2.5527e+01 SSIM: 8.6713e-01
# UNET UIEB
# Validation # PSNR: 2.4773e+01 SSIM: 9.1547e-01

# dpm LSUI
# Validation # PSNR: 2.2011e+01
# Validation # SSIM: 8.4009e-01

# Validation # PSNR: 2.2131e+01
# Validation # SSIM: 8.3792e-01

