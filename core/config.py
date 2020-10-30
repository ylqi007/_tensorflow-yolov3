#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/21/20 1:33 PM
# @Author  : ylqi007
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
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.UPSAMPLE_METHOD = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt"


# =============================================================================
# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "./data/dataset/voc_train.txt"
__C.TRAIN.BATCH_SIZE = 6
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_coco_demo.ckpt"

# =============================================================================
# Test options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "./data/dataset/voc_test.txt"
__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 544
__C.TEST.DATA_AUG = False





