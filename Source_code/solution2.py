import sys
sys.path.append('./')

import random
import cv2
import numpy as np
from Source_code.utils import SIFT

def load_img(imgpath, is_gray=False):
    if is_gray:
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(imgpath)
    return img


class Stitching(object):
    def __init__(self, imgpath1, imgpath2, savename):
        self.img1 = load_img(imgpath1, is_gray=False)
        self.img2 = load_img(imgpath2, is_gray=False)
        self.img1_gray = load_img(imgpath1, is_gray=True)
        self.img2_gray = load_img(imgpath2, is_gray=True)
        self.savename = savename

        self.bfm_matcher = cv2.BFMatcher_create(cv2.NORM_L1)
        self.better_threshold = 0.6
        self.min_match_num = 10

        self.colors = [(225, 43, 40), (47, 99, 219), (45, 155, 66)]
        self.radius = 2

    def run(self):
        keypoints1, descriptors1 = SIFT.computeKeypointsAndDescriptors(self.img1_gray)
        keypoints2, descriptors2 = SIFT.computeKeypointsAndDescriptors(self.img2_gray)

        matches = self.bfm_matcher.knnMatch(descriptors1, descriptors2, k=2)
        better_matches = [item[0] for item in matches if item[0].distance < self.better_threshold * item[1].distance]

        match_img = self.draw_matches(keypoints1, keypoints2, better_matches[:30])
        cv2.imwrite(f'./Figures/output/{self.savename}-match.png', match_img)
        print(f'>> Match successfully')

        if len(better_matches) > self.min_match_num:
            src_points = np.float32([keypoints1[item.queryIdx].pt for item in better_matches]).reshape(-1, 1, 2)
            dst_points = np.float32([keypoints2[item.trainIdx].pt for item in better_matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            self.warpImages(M)
        else:
            raise NotImplementedError

    def draw_matches(self, keypoints1, keypoints2, matches):
        r1, c1 = self.img1_gray.shape[:2]
        r2, c2 = self.img2_gray.shape[:2]

        match_img = np.full((max([r1, r2]), c1 + c2 + 10, 3), 255).astype('uint8')
        match_img[:r1, :c1, :] = np.dstack([self.img1[:, :, 0], self.img1[:, :, 1], self.img1[:, :, 2]])
        match_img[:r2, c1 + 10:c1 + c2 + 10, :] = np.dstack([self.img2[:, :, 0], self.img2[:, :, 1], self.img2[:, :, 2]])

        for match in matches:
            color = random.choice(self.colors)
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            x1, y1 = keypoints1[img1_idx].pt
            x2, y2 = keypoints2[img2_idx].pt

            cv2.circle(match_img, (int(x1), int(y1)), self.radius, color, 1)
            cv2.circle(match_img, (int(x2) + c1 + 10, int(y2)), self.radius, color, 1)
            cv2.line(match_img, (int(x1), int(y1)), (int(x2) + c1 + 10, int(y2)), color, 1)

        return match_img

    def warpImages(self, M):
        height1, width1 = self.img2.shape[:2]
        height2, width2 = self.img1.shape[:2]

        I1_corners = [[0, 0], [0, height1], [width1, height1], [width1, 0]]
        I2_corners = [[0, 0], [0, height2], [width2, height2], [width2, 0]]

        p1 = np.float32(I1_corners).reshape(-1, 1, 2)
        p2_corners = np.float32(I2_corners).reshape(-1, 1, 2)
        p2 = cv2.perspectiveTransform(p2_corners, M)

        all_points = np.concatenate((p1, p2), axis=0)
        x_min, y_min = np.int32(np.min(all_points, axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(np.max(all_points, axis=0).ravel() + 0.5)

        coefficient = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        result = cv2.warpPerspective(self.img1, coefficient.dot(M), (x_max - x_min, y_max - y_min))
        result[-y_min:height1 + -y_min, -x_min:width1 + -x_min] = self.img2

        cv2.imwrite(f'./Figures/output/{self.savename}-final.png', result)


if __name__ == "__main__":
    stitcher = Stitching('./Figures/source/city1.png', './Figures/source/city2.png', 'city')
    # stitcher = Stitching('./Figures/source/mc1/mc1-1.png', './Figures/source/mc1/mc1-2.png', 'mc1')
    # stitcher = Stitching('./Figures/source/mc2/mc2-1.png', './Figures/source/mc2/mc2-2.png', 'mc2')
    # stitcher = Stitching('./Figures/source/village/village.jpg', './Figures/source/village/village.jpg', 'village')
    stitcher.run()
