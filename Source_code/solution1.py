# -*- coding: utf-8 -*-
# @Author : Hcyang-NULL
# @Time   : 2021/3/29 2:36 下午
# - - - - - - - - - - -

import cv2
import numpy as np
from matplotlib import pyplot as plt


def _topN(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


class Defogger(object):
    def __init__(self, img):
        self.I = img
        self.A = None
        self.t = None
        self.J = None
        self.I_dark = None
        self.I_dark_norm = None

        # Dark channel filter size
        self.dark_kernel_r = 5
        # the lower bound of the global atmospheric transmission
        self.t0 = 0.1
        # fix parameter
        self.w = 0.95

    def _dark_channel(self, norm=False):
        if norm:
            I = np.array(self.I / self.A)
        else:
            I = np.array(self.I)

        h, w, c = I.shape
        I = np.min(I, axis=2)

        dr = self.dark_kernel_r
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dr + 1, 2 * dr + 1))

        I_padding = np.full((h + 2 * dr, w + 2 * dr), 255).astype('float')
        I_padding[dr:(dr + h), dr:(dr + w)] = I
        I_dark_padding = np.zeros_like(I_padding)

        for i in range(dr, dr + h):
            for j in range(dr, dr + w):
                local_field = I_padding[(i - dr):(i + dr + 1), (j - dr):(j + dr + 1)]
                local_min = np.min(local_field[ellipse == 1])
                I_dark_padding[i, j] = local_min

        I_dark = I_dark_padding[dr:dr + h, dr:dr + w]

        if norm:
            self.I_dark_norm = I_dark
        else:
            self.I_dark = I_dark

    def _estimate_A(self):
        h, w = self.I_dark.shape
        top_pixels = h * w // 1000

        bright_index = _topN(self.I_dark, top_pixels)

        A = np.zeros((3,))
        A[0] = np.max(self.I[bright_index][0])
        A[1] = np.max(self.I[bright_index][1])
        A[2] = np.max(self.I[bright_index][2])

        self.A = A

    def _estimate_t(self):
        t = 1 - self.w * self.I_dark_norm
        self.t = t

    def _recover_J(self):
        t = cv2.max(self.t, self.t0)
        J = np.zeros_like(self.I)
        for c in range(self.I.shape[2]):
            J[:, :, c] = (self.I[:, :, c] - self.A[c]) / t + self.A[c]
        self.J = J.astype('uint8')

    def defogging(self):
        self._dark_channel(norm=False)
        self._estimate_A()
        self._dark_channel(norm=True)
        self._estimate_t()
        self._recover_J()
        return self.J

def add_fog(ori_img):
    ori_img = ori_img / 255
    A = np.random.uniform(0.6, 0.95)
    t = np.random.uniform(0.2, 0.95)
    foggy_img = ori_img * t + A * (1 - t)
    return foggy_img

def main():
    fog_img = cv2.imread('./Figures/source/Ex_ColorEnhance.png')
    defogger = Defogger(fog_img)
    defogging_img = defogger.defogging()

    origin_img = cv2.imread('./Figures/source/Tam_clear.jpg')
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    foggy_img = add_fog(origin_img)

    fig = plt.figure(figsize=(13, 10))
    plt.subplot(221)
    plt.imshow(fog_img)
    plt.axis('off')
    plt.title('Fog Image')
    plt.subplot(222)
    plt.imshow(defogging_img)
    plt.axis('off')
    plt.title('Defogging Image')
    plt.subplot(223)
    plt.imshow(origin_img)
    plt.axis('off')
    plt.title('Origin Image')
    plt.subplot(224)
    plt.imshow(foggy_img)
    plt.axis('off')
    plt.title('Add Fog Image')
    # plt.savefig('./Figures/output/fog_defog.png', dpi=400)
    plt.show()


if __name__ == '__main__':
    main()
