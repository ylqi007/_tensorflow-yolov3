#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/21/20 12:54 PM
# @Author  : Shark
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg

# TODO: random
# TODO: ','.join()
# TODO: lambda


class Dataset(object):
    """ Implement Dataset class. """
    def __init__(self, dataset_type):
        """
        Initialize a dataset for training or testing.
        :param dataset_type:
        """
        # TODO, Initialize the dataset class
        self.annot_path     = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes    = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.batch_size     = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug       = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes  = cfg.TRAIN.INPUT_SIZE
        self.strides            = np.array(cfg.YOLO.STRIDES)
        self.classes            = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes        = len(self.classes)
        self.anchors            = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale   = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides     # floor division

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))  # (batch_size, W, H, 3)

            # label
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))
            # bbox
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batches:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:   # self.num_samples is the total num of samples
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)


    def __len__(self):
        return self.num_batches

    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            _annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(_annotations)
        return _annotations

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape       # (y, x, c), i.e. (h, w, c)
            image = image[:, ::-1, :]   # 水平翻转, See my_logs.md
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            # max_bbox = [xmin, ymin, xmax, ymax], xmin is the minimum value of all xmins
            max_l_trans = max_bbox[0]   # the max value can transfer left
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            # cropped [xmin, ymin, xmax, ymax] = [crop_xmin, crop_ymin, crop_xmax, crop_ymax]
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(0, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(0, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]     # row/col, i.e. y/x
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))  # shift horizontally
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))  # shift vertically

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return image, bboxes

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("{} does not exits.".format(image_path))
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))
        # line 151 is the final code, 152 for test
        # image, bboxes = utils.image_preprocess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        image, bboxes = utils.image_preprocess(np.copy(image), [self.train_input_sizes[-1], self.train_input_sizes[-1]], np.copy(bboxes))   # for test
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):
        # TODO
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)
        # print("boxes1:\n", boxes1)
        # print("anchors:\n", boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]
        # print("boxes1_area: ", boxes1_area)
        # print("anchor_area: ", boxes2_area)

        # xywh -> xyxy
        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
        # print("-------")
        # print("boxes_xyxy:\n", boxes1)
        # print("anchor_xyxy:\n", boxes2)
        # print('=======')
        # print("left up:\n", left_up)
        # print("right down:\n", right_down)
        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        print("IOU:\n", inter_area / union_area)
        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):
        # Def self.train_output_sizes for test here
        # self.train_output_sizes = 512 // np.array([8, 16, 32])
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i],          # [(64, 64, 3, 25), (32, 32, 3, 25), (16, 16, 3, 25)]
                           self.anchor_per_scale, 5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]            # [np.zeros(3,4), np.zeros(3,4), np.zeros(3,4)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:             # process each truth bounding box separately
            bbox_coor = bbox[:4]        # xmin, ymin, xmax, ymax
            bbox_class_ind = bbox[4]    # class id

            # label probability
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # bbox in (x, y, w, h) format. (x,y), the center of the bbox, (w,h), the w and h of the bbox
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # bbox_xywh_scaled: shape(1, 4) / shape(3, 1) ==> shape(3, 4), broadcasting
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis] # scale to (64x64), (32,32), and (16,16)
            # print("$$$$ bbox_xywh: ", bbox_xywh)
            # print("#### bbox_xywh_scaled: \n", bbox_xywh_scaled)
            # print("###########################################")
            iou = []
            exist_positive = False
            for i in range(3):  # strides = [8, 16, 32]
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))  # (3, 4), the xywh of three different scale anchors
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # xc, yc of anchors
                anchors_xywh[:, 2:4] = self.anchors[i]  # w, h of anchors, three different anchors
                # print("$$$$ anchors_xywh: \n", anchors_xywh, "\n")

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
                # print(iou_mask)

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)    # xind, yind is the center of selected anchor

                    label[i][yind, xind, iou_mask, :] = 0           # why yind, xind
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                # print("================== No True ====================")
                # print("np.array(iou):\n", np.array(iou).reshape(-1))

                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1) # the np.array(iou).reshape(-1) should be (9,3), i.e. 9 anchors
                # print(best_anchor_ind)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)  # best_detect = 0, i.e. bbox_xywh[0]
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)  # xcenter,ycenter in scaled images

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh    # bbox_xywh,
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        # print("============= End of dataset.preprocess_true_boxes ==============")
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

