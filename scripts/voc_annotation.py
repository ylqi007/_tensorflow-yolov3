#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Time    : 10/21/20 5:14 PM
# Author  : Shark
# Site    : 
# File    : voc_annotation.py
# Software: PyCharm
# ===========================================================

import os
import argparse
import xml.etree.ElementTree as ET

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']


def convert_voc_annotations(data_path, data_type, anno_path, use_difficult_bbox=True):
    """

    :param data_path:
    :param data_type:
    :param anno_path:
    :param use_difficult_bbox:
    :return:
    """
    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_id = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
            f.write(annotation + '\n')
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/ylqi007/work/DATA/VOC2007")
    # The following default directory is relative path the `_tensorflow-yolov3/`
    parser.add_argument("--train_annotation", default="./data/dataset/voc_train.txt")
    parser.add_argument("--test_annotation", default="./data/dataset/voc_test.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):
        os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):
        os.remove(flags.test_annotation)

    num1 = convert_voc_annotations(os.path.join(flags.data_path, 'train/'), 'trainval', flags.train_annotation, False)
    num3 = convert_voc_annotations(os.path.join(flags.data_path, 'test/'), 'test', flags.test_annotation, False)
    print("=> The number of image for train is: {}\n=> The number of image for test is: {}".format(num1, num3))
