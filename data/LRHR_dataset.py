from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np
import cv2
from matplotlib import pyplot as plt
# from model.utils import categories
def color_filter(image):
    gray_SR = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_SR_gray = np.zeros_like(image)
    img_SR_gray[:, :, 0] = gray_SR
    img_SR_gray[:, :, 1] = gray_SR
    img_SR_gray[:, :, 2] = gray_SR

    result = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 0, 20])
    upper1 = np.array([40, 255, 255])

    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([125, 0, 20])
    upper2 = np.array([179, 255, 255])

    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    full_mask = lower_mask + upper_mask
    # print(np.unique(full_mask))
    neg_full_mask = 255 - full_mask

    result = cv2.bitwise_and(result, result, mask=full_mask)
    result_neg = cv2.bitwise_and(img_SR_gray, img_SR_gray, mask=neg_full_mask)
    return result_neg + result[:, :, ::-1]

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

class LRHRDataset2(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False, label_path=None):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.need_label = False
        if label_path is not None:
            labels = {}
            for line in open(label_path):
                image_name, label = line.split(' ')
                labels[image_name] = int(label.strip())
            self.labels = labels
            self.need_label = True

        if datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}_gray'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            self.style_path = Util.get_paths_from_images(
                '{}/hr_{}_style'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        index_style = np.random.randint(0, self.data_len)

        # img_HR = Image.open(self.hr_path[index]).convert("RGB").resize((128, 128))
        # img_SR = Image.open(self.sr_path[index]).convert("RGB").resize((128, 128))
        # img_style = Image.open(self.style_path[index]).convert("RGB").resize((64, 64))

        img_HR = Image.open(self.hr_path[index]).convert("RGB")
        img_SR = Image.open(self.sr_path[index]).convert("RGB")
        # img_style = Image.open(self.style_path[index]).convert("RGB")
        img_style = Image.open(self.style_path[index_style]).convert("RGB")

        # img_HR = cv2.imread(self.hr_path[index])
        # img_HR = img_HR[:, :, ::-1]
        # img_HR = img_HR.copy()
        # img_SR = cv2.imread(self.sr_path[index])
        # img_style = cv2.imread(self.style_path[index])
        # img_style = img_style[:, :, ::-1]
        # img_style = img_style.copy()
        # img_SR = color_filter(img_SR)
        # img_SR = img_SR.copy()
        if self.need_label:
            label = self.labels[self.hr_path[index]]
        else:
            label = None
        # img_style = Image.open(self.sr_path[index]).convert("RGB")
        if self.need_LR:
            img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR, img_style] = Util.transform_augment(
                [img_LR, img_SR, img_HR, img_style], split=self.split, min_max=(-1, 1))
            if label is None:
                return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'style': img_style, 'Index': index}
            else:
                return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'style': img_style, 'Index': index, 'label': label}
        else:
            [img_SR, img_HR, img_style] = Util.transform_augment(
                [img_SR, img_HR, img_style], split=self.split, min_max=(-1, 1))
            if label is None:
                return {'HR': img_HR, 'SR': img_SR, 'style': img_style, 'Index': index}
            else:
                return {'HR': img_HR, 'SR': img_SR, 'style': img_style, 'Index': index, 'label': label}
