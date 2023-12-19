import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob
import core.logger as Logger
import argparse
from model.ddpm_trans_modules.unet import UNet
from model.ddpm_trans_modules.unet_backup import DiT
from model.ddpm_trans_modules import diffusion
from model.utils import load_part_of_model3, load_part_of_model4

# imgs = os.listdir('results')
#
# for i, name in enumerate(imgs):
#     name_, suffix = os.path.splitext(name)
#     ori = Image.open(os.path.join('results', name))
#     new = ori.save(os.path.join('results2', '0_' + str(i+1) + '_sr.png'))
# num = 0
# path = 'experiments_val/sr_ffhq_230926_165315/results'
# save_path = 'experiments_val/sr_ffhq_231107_103219/hr_128'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# real_names = list(glob.glob('{}/*_sr.png'.format(path)))
# for name in real_names:
#     ori = Image.open(os.path.join(name))
#     s = '%05d' % num
#     new = ori.save(os.path.join(save_path, str(s) + '.png'))
#     num = num + 1

import torch
import clip
from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# print(text.shape)
# with torch.no_grad():
#     # image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     print(text_features.shape)

    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

# model = DiT(depth=12, in_channels=4, hidden_size=384, patch_size=4, num_heads=6, input_size=128)
# src_model_path = 'experiments_train/sr_ffhq_231113_172326/checkpoint/I550000_E1951_gen.pth'
#
# netG = diffusion.GaussianDiffusion(
#         model,
#         image_size=128,
#         channels=4,
#         loss_type='l1',    # L1 or L2
#         conditional=True,
# )
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--config', type=str, default='config/underwater.json',
#                     help='JSON file for configuration')
# parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
# parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
# parser.add_argument('-debug', '-d', action='store_true')
# parser.add_argument('-enable_wandb', action='store_true')
# parser.add_argument('-log_infer', action='store_true')

# parse configs
# args = parser.parse_args()
# opt = Logger.parse(args)
# # Convert to NoneDict, which return None for missing key.
# opt = Logger.dict_to_nonedict(opt)
#
# netG.set_new_noise_schedule(
#         opt['model']['beta_schedule'][opt['phase']], device='cpu')
# netG.set_loss('cpu')
# load_part_of_model4(netG, src_model_path)

labels = {0: 'fish', 1: 'marine life', 2: 'coral', 3: 'rocky', 4: 'diving people',
                  5: 'deep see scenes', 6: 'wreckage', 7: 'sculpture', 8: 'underwater caves', 9: 'underwater stuff'}


def generate_segmentation(root, label_path):
    for line in open(label_path):
        image_name, label = line.split(' ')
        input_txt = labels[int(label.strip())]
        image = Image.open(root + '/' + image_name).convert('RGB')

def calc_mean_std(input, eps=1e-5):
    batch_size, channels = input.shape[:2]

    reshaped = input.view(batch_size, channels, -1) # Reshape channel wise
    mean = torch.mean(reshaped, dim = 2).view(batch_size, channels, 1, 1) # Calculat mean and reshape
    std = torch.sqrt(torch.var(reshaped, dim=2)+eps).view(batch_size, channels, 1, 1) # Calculate variance, add epsilon (avoid 0 division),
                                                                                # calculate std and reshape
    return mean, std

def AdaIn(content, style):
    assert content.shape[:2] == style.shape[:2] # Only first two dim, such that different image sizes is possible
    batch_size, n_channels = content.shape[:2]
    mean_content, std_content = calc_mean_std(content)
    mean_style, std_style = calc_mean_std(style)

    output = std_style*((content - mean_content) / (std_content)) + mean_style # Normalise, then modify mean and std
    return output

if __name__ == '__main__':

    root = '/home/ty/code/DM_underwater/dataset/water_train_16_256/sr_16_256'
    style_root = '/home/ty/code/DM_underwater/dataset/water_train_16_128/hr_128_style'
    label_path = '/home/ty/code/DM_underwater/dataset/water_train_16_128/label.txt'
    # generate_segmentation(root, label_path)

    img = cv2.imread(os.path.join(root, '00001.png'), 1)
    style_img = cv2.imread(os.path.join(style_root, '00003.png'), 1)

    # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # style_img2 = cv2.cvtColor(style_img, cv2.COLOR_BGR2LAB)

    # input = torch.from_numpy(img2).type(torch.FloatTensor)
    # style = torch.from_numpy(style_img2).type(torch.FloatTensor)
    # print(input.shape)
    # input = torch.permute(input, (2, 0, 1))
    # input = input[None, :]

    # style = torch.permute(style, (2, 0, 1))
    # style = style[None, :]

    # output = AdaIn(input, style)
    # output = output.data.cpu().numpy()
    # print(output.shape)
    style = cv2.resize(style_img, (256, 256))
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    style_lab = cv2.cvtColor(style, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    # print(np.unique(a))
    la = np.mean(img, axis=2)
    print(l.shape)
    l_s, a_s, b_s = cv2.split(style_lab)
    da = (a - np.mean(a)) / np.std(a) * np.std(a_s) + np.mean(a_s)
    db = (b - np.mean(b)) / np.std(b) * np.std(b_s) + np.mean(b_s)
    # print(np.unique(da))
    da = da.astype(np.uint8)
    db = db.astype(np.uint8)
    lab_image = np.stack([l, da, db])
    output = np.transpose(lab_image, (1, 2, 0))
    output = output.astype(np.uint8)
    print(output.shape)
    output = cv2.cvtColor(output, cv2.COLOR_LAB2BGR)
    plt.subplot(1, 3, 1)
    plt.imshow(img[:, :, ::-1])
    plt.subplot(1, 3, 2)
    plt.imshow(style_img[:, :, ::-1])
    plt.subplot(1, 3, 3)
    plt.imshow(output[:, :, ::-1])
    plt.show()




