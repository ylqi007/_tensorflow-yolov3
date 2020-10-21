#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/21/20 1:33 PM
# @Author  : Shark
# @Site    : 
# @File    : config.py
# @Software: PyCharm

from easydict import EasyDict as edict

# Consumers can get config by: from config import cfg
__C = edict()
cfg = __C

# =============================================================================
# YOLO options, set the class name
__C.YOLO = edict()

__C.YOLO.CLASSES = "./data/classes/voc.names"
__C.YOLO.ANCHORS = "./data/anchors/baseline_anchors.txt"
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3


# =============================================================================
# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "./data/dataset/voc_train.txt"
__C.TRAIN.BATCH_SIZE = 6
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG = True

# =============================================================================
# Test options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "./data/dataset/voc_test.txt"
__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 544
__C.TEST.DATA_AUG = False





