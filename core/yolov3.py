#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Time    : 10/25/20 8:17 PM
# Author  : ylqi007
# Site    : 
# File    : yolov3.py
# Software: PyCharm
# ===========================================================

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.backbone as backbone
from core.config import cfg


class YOLOV3(object):
    """ Implement tensorflow yolov3. """
    def __init__(self, input_data, trainable):
        self.trainable          = trainable
        self.classes            = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes        = len(self.classes)
        self.strides            = np.array(cfg.YOLO.STRIDES)
        self.anchors            = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale   = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh    = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method    = cfg.YOLO.UPSAMPLE_METHOD

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_network(input_data)
        except:
            raise NotImplementedError("Cannot build up yolov3 network!")

        with tf.variable_scope("pred_sbbox"):
            pass

        with tf.variable_scope("pred_mbbox"):
            pass

        with tf.variable_scope("pred_lbbox"):
            pass

    def __build_network(self, input_data):
        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)
        pass

    def decode(self, conv_output, anchors, stride):
        pass

    def focal(self, target, actual, alpha=1, gamma=2):
        pass

    def bbox_giou(self, boxes1, boxes2):
        pass

    def bbox_iou(self, boxes1, boxes2):
        pass

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        pass

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        pass