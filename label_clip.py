import os

import cv2
from PIL import Image
import glob
import torch
import clip
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

categories = ['fish', 'marine life', 'coral', 'rock', 'diving', 'deep see',
                      'wreckage', 'sculpture', 'caves', 'underwater stuff']

categories_red_color = ['red', 'green', 'blue']
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
text = clip.tokenize(categories).to(device)
color_text = clip.tokenize(categories_red_color).to(device)

def generate_clip_label():
    path = 'dataset/water_val_16_128/hr_128'
    save_path = 'dataset/water_val_16_128'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    real_names = list(glob.glob('{}/*.png'.format(path)))
    real_names.sort()
    file = open(os.path.join(save_path, 'label.txt'), 'w')
    with torch.no_grad():
        for name in real_names:
            # print(name)
            image_name = name.split('/')[-1]
            image = preprocess(Image.open(name)).unsqueeze(0).to(device)
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            category = categories[np.argmax(probs)]
            print(image_name, '-----', category, '----', np.max(probs))
            file.writelines(os.path.join('dataset/water_val_16_128/hr_128', image_name) + ' ' + str(categories.index(category)) + '\n')

        file.close()

def judge_color_clip(image):

    with torch.no_grad():
        image = preprocess(image).unsqueeze(0).to(device)
        # image_features = model.encode_image(image)
        # text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, color_text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        color_level = categories_red_color[np.argmax(probs)]
        print(color_level, '----', np.max(probs))

def compute_semantic_dis(image, txt):

    with torch.no_grad():
        image = preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize(txt).to(device)
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        # logits_per_image, logits_per_text = model(image, color_text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # color_level = categories_red_color[np.argmax(probs)]
        dis = cos(image_features, text_features)
        # print('distance between image and ', txt, ' =', dis)
        return dis.data.cpu().numpy()

def compute_hist_dis(image, anchor, color):
    chans = cv2.split(image)
    chans_anchor = cv2.split(anchor)
    if color == 'red':
        chan = chans[2]
        chan_anchor = chans_anchor[2]
    elif color == 'green':
        chan = chans[1]
        chan_anchor = chans_anchor[1]
    else:
        chan = chans[0]
        chan_anchor = chans_anchor[0]

    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_anchor = cv2.calcHist([chan_anchor], [0], None, [256], [0, 256])
    cv2.normalize(hist_anchor, hist_anchor, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    metric_val = cv2.compareHist(hist, hist_anchor, cv2.HISTCMP_CHISQR)
    return metric_val
def test_clip_loss():
    path = 'dataset/water_train_16_128/sr_16_128/00151.png'
    path2 = 'dataset/water_train_16_128/hr_128/00151.png'
    src = preprocess(Image.open(path)).unsqueeze(0).to(device)
    target = preprocess(Image.open(path2)).unsqueeze(0).to(device)
    src_txt = 'diving in underwater scenes'
    target_txt = 'diving without underwater'

    text = clip.tokenize(src_txt).to(device)
    text2 = clip.tokenize(target_txt).to(device)

    src_image_features = model.encode_image(src)
    target_image_features = model.encode_image(target)
    src_text_features = model.encode_text(text)
    target_text_features = model.encode_text(text2)

    print(src_text_features.shape, '--', src_image_features.shape)

    src_image_features /= src_image_features.norm(dim=-1, keepdim=True)
    target_image_features /= target_image_features.norm(dim=-1, keepdim=True)
    src_text_features /= src_text_features.norm(dim=-1, keepdim=True)
    target_text_features /= target_text_features.norm(dim=-1, keepdim=True)

    src_fuse_features = torch.cat([src_image_features, src_text_features], dim=-1)
    target_fuse_features = torch.cat([target_image_features, target_text_features], dim=-1)

    direction = target_image_features - src_image_features
    direction /= direction.norm(dim=-1, keepdim=True)

    text_direction = target_text_features - src_text_features
    text_direction /= text_direction.norm(dim=-1, keepdim=True)

    value = torch.nn.CosineSimilarity()(direction, text_direction)
    value2 = torch.nn.CosineSimilarity()(src_fuse_features, target_fuse_features)

    print('cosine distance:', value)
    print('cosine distance2:', value2)
    # print(1 - value)

if __name__ == '__main__':
    # test_clip_loss()
    # generate_clip_label()
    # image = Image.open('./b.jpg').convert('RGB')
    # image.resize((128, 128))
    # image2 = Image.open('./a.jpg').convert('RGB')
    image = cv2.imread('./momiji.jpeg')
    image2 = cv2.imread('./a.jpg')
    image3 = cv2.imread('./b.jpg')
    print(compute_hist_dis(image2, image, 'red'))
    print(compute_hist_dis(image3, image, 'red'))
    # judge_color_clip(image)
    # judge_color_clip(image2)
    # compute_semantic_dis(image, 'red style')
    # compute_semantic_dis(image2, 'red style')
    # chans = cv2.split(image)
    # chans2 = cv2.split(image2)
    #
    # colors = ('b', 'g', 'r')
    #
    # for i, (chan, color) in enumerate(zip(chans, colors)):
    #     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    #     hist2 = cv2.calcHist([chans2[i]], [0], None, [256], [0, 256])
    #     plt.plot(hist, color=color)
    #     plt.plot(hist2, color=color, linestyle='dashed')
    #     plt.xlim([0, 256])
    #
    # plt.show()






