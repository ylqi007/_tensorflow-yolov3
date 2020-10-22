# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import copy
from core.dataset import Dataset


def cv2_img_read_test():
    img = cv2.imread('/home/ylqi007/work/DATA/VOC2007/test/JPEGImages/003982.jpg', 1)
    print(type(img), img.shape)     # shape = (y, x, channel) = (h, w, c) = (333, 500, 3)
    flip_img = np.copy(img)
    flip_img = flip_img[:, ::-1, :]
    cv2.imshow('original', img)
    cv2.imshow('reversed', flip_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def annotation_flip_test():
    dataset = Dataset('train')
    anno = '/home/ylqi007/work/DATA/VOC2007/test/JPEGImages/004538.jpg 215,35,308,224,14 41,77,403,367,12'
    image, bboxes = dataset.parse_annotation(annotation=anno)
    flip_img, flip_bboxes = np.copy(image), np.copy(bboxes)
    _, w, _ = flip_img.shape
    flip_img = flip_img[:, ::-1, :]
    flip_bboxes[:, [0, 2]] = w - flip_bboxes[:, [2, 0]]

    print("========== original box ==========")
    # show_image('original', image)
    show_image('original_with_bbox', image, bboxes)

    print("========== flipped box ==========")
    show_image('flipped_with_bbox', flip_img, flip_bboxes)
    # cv2.rectangle(flip_img, (flip_bboxes[0][0], flip_bboxes[0][1]), (flip_bboxes[0][2], flip_bboxes[0][3]), (0, 255, 0), 2)
    # cv2.imshow("flipped_with_bboxes", flip_img)
    # while cv2.waitKey(100) != 27:
    #     if cv2.getWindowProperty("flipped_with_bboxes", cv2.WND_PROP_VISIBLE) <= 0:
    #         break
    # cv2.destroyAllWindows()


def show_image(img_name, img, bboxes=None):
    if bboxes is None:
        cv2.imshow(img_name, img)
    else:
        for bbox in bboxes:
            print('bbox: ', bbox)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imshow(img_name, img)
        cv2.imshow(img_name, img)
    while cv2.waitKey(100) != 27:
        if cv2.getWindowProperty(img_name, cv2.WND_PROP_VISIBLE) <= 0:
            break
    cv2.destroyAllWindows()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Dataset test
    # dataset = Dataset('test')
    # annotations = dataset.annotations
    # print(type(annotations))
    # for anno in annotations:
    #     print(anno)

    # cv2 test
    # cv2_img_read_test()

    # Dataset test
    annotation_flip_test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
