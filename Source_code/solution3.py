# -*- coding: utf-8 -*-
# @Author : Hcyang-NULL
# @Time   : 2021/3/30 10:31 下午
# - - - - - - - - - - -

import cv2
import numpy as np
from matplotlib import pyplot as plt


def _load_img(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    return img


class SpaceTransform(object):
    def __init__(self):
        self.coefficient = {
            'rgb2xyz': [
                [0.5141, 0.3239, 0.1604],
                [0.2651, 0.6702, 0.0641],
                [0.0241, 0.1228, 0.8444]
            ],
            'xyz2lms': [
                [0.3897, 0.689, -0.0787],
                [-0.2298, 1.1834, 0.0464],
                [0., 0., 1.]
            ],
            'rgb2lms': [
                [0.3811, 0.5783, 0.0402],
                [0.1967, 0.7244, 0.0782],
                [0.0241, 0.1228, 0.8444]
            ],
            'lms2lab': [
                [0.57735027, 0.57735027, 0.57735027],
                [0.40824829, 0.40824829, -0.81649658],
                [0.70710678, -0.70710678, 0.]
            ],
            'lab2lms': [
                [0.57735027, 0.40824829, 0.70710678],
                [0.57735027, 0.40824829, -0.70710678],
                [0.57735027, -0.81649658, 0.]
            ],
            'lms2rgb': [
                [4.4679, -3.5873, 0.1193],
                [-1.2186, 2.3809, -0.1624],
                [0.0497, -0.2439, 1.2045]
            ]
        }

    def space_transform(self, img, ctype):
        coeff = np.array(self.coefficient[ctype])
        origin_shape = img.shape
        new_img = coeff @ img.reshape(-1, 3).T
        return new_img.T.reshape(origin_shape)

    def style_transform(self, src, dst):
        src_mean = src.reshape(-1, 3).mean(axis=0)
        src_std = src.reshape(-1, 3).std(axis=0)
        dst_mean = dst.reshape(-1, 3).mean(axis=0)
        dst_std = dst.reshape(-1, 3).std(axis=0)

        src_gap = src - src_mean
        lab_img = dst_std / src_std * src_gap + dst_mean

        lms_img = self.space_transform(lab_img, 'lab2lms')
        lms_img = np.power(10, lms_img)
        rgb_img = self.space_transform(lms_img, 'lms2rgb')

        return lab_img, rgb_img


class ColorTransfer(object):
    def __init__(self, src, dst, show=True):
        self.src_rgb_img = _load_img(src)
        self.dst_rgb_img = _load_img(dst)
        self.space_trans = SpaceTransform()
        self.epsilon = 1e-10
        self.show = show

        self.src_lab_img = None
        self.dst_lab_img = None

        self._prepare()

    def plot(self, savename=''):
        if self.show:
            plt.show()
        else:
            plt.savefig(savename, dpi=400)
        plt.close('all')

    def rgb2lab(self, rgb_img):
        lms_img = self.space_trans.space_transform(rgb_img, 'rgb2lms')
        lms_img += self.epsilon
        lms_img = np.log10(lms_img)
        lab_img = self.space_trans.space_transform(lms_img, 'lms2lab')
        return lab_img

    def lab2rgb(self, lab_img):
        lms_img = self.space_trans.space_transform(lab_img, 'lab2lms')
        lms_img = np.power(10, lms_img)
        rgb_img = self.space_trans.space_transform(lms_img, 'lms2rgb')
        return rgb_img

    def _prepare(self):
        self.src_lab_img = self.rgb2lab(self.src_rgb_img)
        self.dst_lab_img = self.rgb2lab(self.dst_rgb_img)

    def task1(self):
        fig = plt.figure(figsize=(13, 10))
        plt.subplot(121)
        plt.title('Source Image (LAB)')
        plt.imshow(self.src_lab_img)
        plt.axis('off')
        plt.subplot(122)
        plt.title('Dest Image (LAB)')
        plt.imshow(self.dst_lab_img)
        plt.axis('off')
        self.plot('./Figures/output/task1.png')

    def task2(self):
        fig = plt.figure(figsize=(17, 10))
        plt.subplot(131)
        plt.title('L Component')
        plt.imshow(self.src_lab_img[:, :, 0], cmap='gray')
        plt.axis('off')
        plt.subplot(132)
        plt.title('Alpha Component')
        plt.imshow(self.src_lab_img[:, :, 1], cmap='gray')
        plt.axis('off')
        plt.subplot(133)
        plt.title('Beta Component')
        plt.imshow(self.src_lab_img[:, :, 2], cmap='gray')
        plt.axis('off')
        self.plot('./Figures/output/task2.png')

    def task3(self):
        lab_img, transfer_img = self.space_trans.style_transform(self.src_lab_img, self.dst_lab_img)
        fig = plt.figure(figsize=(13, 10))
        plt.subplot(121)
        plt.title('LAB Image')
        plt.imshow(lab_img)
        plt.axis('off')
        plt.subplot(122)
        plt.title('RGB Transfered Image')
        plt.imshow(transfer_img)
        plt.axis('off')
        self.plot('./Figures/output/task3.png')

    def task4(self, offset):
        lab_img, transfer_img = self.space_trans.style_transform(self.src_lab_img, self.dst_lab_img)
        fig = plt.figure(figsize=(17, 10))
        plt.subplot(131)
        plt.title('Original Image')
        plt.imshow(self.src_rgb_img)
        plt.axis('off')
        plt.subplot(132)
        plt.title('Target Image')
        plt.imshow(self.dst_rgb_img)
        plt.axis('off')
        plt.subplot(133)
        plt.title('Color Transfer Image')
        plt.imshow(transfer_img)
        plt.axis('off')
        self.plot(f'./Figures/output/task{4 + offset}.png')


show = True

color_transfer = ColorTransfer('./Figures/source/houses.bmp', './Figures/source/hats.bmp', show=show)
color_transfer.task1()
color_transfer.task2()
color_transfer.task3()
color_transfer.task4(0)

color_transfer = ColorTransfer('./Figures/source/Starry_night.png', './Figures/source/Mountains.png', show=show)
color_transfer.task4(1)

color_transfer = ColorTransfer('./Figures/source/Mountains.png', './Figures/source/Starry_night.png', show=show)
color_transfer.task4(2)

color_transfer = ColorTransfer('./Figures/source/cy_src.png', './Figures/source/cy_dst.png', show=show)
color_transfer.task4(3)
