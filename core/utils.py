#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Time    : 10/21/20 2:24 PM
# Author  : Shark
# Site    : 
# File    : utils.py
# Software: PyCharm
# ===========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_class_names(class_file_name):
    """ load class name from a file.
    id is the index, i.e. the line number.

    """
    names = {}
    with open(class_file_name, 'r') as f:
        for idx, name in enumerate(f):
            names[idx] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    """ Load the anchors from a file. """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preprocess(image, target_size, gt_bboxes=None):
    # plt.imshow(image, aspect="auto")
    # plt.title("image_process: original image(BGR)")
    # plt.show()
    # cv2.imshow("image_preprocess: original image(BGR)", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)   # (1, 255) --> (1.0, 255.0)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (1, 255) --> (1.0, 255.0)
    # plt.imshow(image, aspect="auto")
    # plt.title("image_process: converted image(RGB)")
    # plt.show()
    # cv2.imshow("image_preprocess: converted image(RGB)", image)

    # while cv2.waitKey(100) != 27:
    #     if cv2.getWindowProperty("image_preprocess: original image(BGR)", cv2.WND_PROP_VISIBLE) <= 0:
    #         break
    # cv2.destroyAllWindows()
    # while cv2.waitKey(100) != 27:
    #     if cv2.getWindowProperty("image_preprocess: converted image(RGB)", cv2.WND_PROP_VISIBLE) <= 0:
    #         break
    # cv2.destroyAllWindows()

    ih, iw = target_size
    h, w, _ = image.shape
    scale = min(iw / w, ih / h) # 500 / 200 = 2.5, 500/300 = 1.67
    nw, nh = int(scale * w), int(scale * h) # new width and new height
    image_resized = cv2.resize(image, (nw, nh)) # Zoom and shrink the image by minimum scale

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh: dh+nh, dw: dw+nw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_bboxes is None:
        return image_paded
    else:
        gt_bboxes[:, [0, 2]] = gt_bboxes[:, [0, 2]] * scale + dw
        gt_bboxes[:, [1, 3]] = gt_bboxes[:, [1, 3]] * scale + dh
        return image_paded, gt_bboxes


if __name__ == "__main__":
    _names = read_class_names("../data/classes/voc.names")
    print(_names)

    _anchors = get_anchors("../data/anchors/baseline_anchors.txt")
    print(_anchors)
