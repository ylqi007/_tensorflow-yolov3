#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/21/20 12:54 PM
# @Author  : Shark
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

import numpy as np
import core.utils as utils
from core.config import cfg


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
        # TODO
        pass

    def __len__(self):
        # TODO
        pass

    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            _annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(_annotations)
        return _annotations

    def random_crop(self, image, bboxes):
        # TODO
        pass

    def random_translate(self, image, bboxes):
        # TODO
        pass

    def parse_annotation(self, annotation):
        # TODO
        pass

    def bbox_iou(self, boxes1, boxes2):
        # TODO
        pass

    def preprocess_true_boxes(self, bboxes):
        # TODO
        pass

