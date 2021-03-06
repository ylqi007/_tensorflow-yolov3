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
    # anno = '/home/ylqi007/work/DATA/VOC2007/test/JPEGImages/007741.jpg 45,254,67,289,2 87,124,119,152,2 66,108,100,121,2 242,66,295,113,2 150,144,192,192,2 221,178,316,227,2 316,198,416,253,2 354,149,390,174,2 253,174,299,199,2 312,75,338,91,2 207,78,235,95,2'
    image, bboxes = dataset.parse_annotation(annotation=anno)
    image = image.copy()    # https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
    flip_img, flip_bboxes = np.copy(image), np.copy(bboxes)
    _, w, _ = flip_img.shape
    flip_img = flip_img[:, ::-1, :]
    flip_bboxes[:, [0, 2]] = w - flip_bboxes[:, [2, 0]]
    flip_img = flip_img.copy()

    print("========== original box ==========")
    # show_image('original', image)
    show_image('original_with_bbox', image, bboxes)

    print("========== flipped box ==========")
    show_image('flipped_with_bbox', flip_img, flip_bboxes)


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


def _test_preprocess_true_boxes():
    dataset = Dataset('train')
    # anno = '/home/ylqi007/work/DATA/VOC2007/test/JPEGImages/004538.jpg 215,35,308,224,14 41,77,403,367,12'
    anno = '/home/ylqi007/work/DATA/VOC2007/test/JPEGImages/004538.jpg 215,35,308,224,14'
    image, bboxes = dataset.parse_annotation(annotation=anno)
    print("bboxes after dataset.parse_annotation()\n", bboxes)
    dataset.preprocess_true_boxes(bboxes)


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
    # annotation_flip_test()

    # Dataset.preprocess_true_boxes() test
    _test_preprocess_true_boxes()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
