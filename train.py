#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===========================================================
# Author      : ylqi007
# Time        : 2020-10-28 10:00 AM
# File        : train.py
# Software    : PyCharm
# Description : 
# ===========================================================
import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_logdir = "./data/log/train"
        self.trainset = Dataset('train')
        self.testset = Dataset('test')
        self.steps_per_period = len(self.trainset)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.name_scope('default_input'):
            self.input_data     = tf.placeholder(dtype=tf.float32, name='input_data')   # input img
            self.label_sbbox    = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox    = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox    = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes   = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes   = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes   = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable      = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('define_loss'):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                self.label_sbbox, self.label_mbbox, self.label_lbbox,
                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False,
                                           name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                       dtype=tf.float64, name='warmup_steps')   # 2 * batches in one epoch
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) *
                                      self.steps_per_period, dtype=tf.float64, name='train_steps')
            # self.learn_rate = tf.cond() #TODO
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (
                            self.learn_rate_init - self.learn_rate_end) *
                                 (1 + tf.cos(
                                     (self.global_step - warmup_steps) / (
                                                 train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope('define_weight_decay'):
            # TODO
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope('define_first_stage_train'):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(
                self.loss, var_list=self.first_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()
            pass

        with tf.name_scope('define_second_stage_train'):
            pass

        with tf.name_scope('loader_and_saver'):
            pass

        with tf.name_scope('summary'):
            pass

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print("=> Restoring weights from: %s ..." % self.initial_weight)
            # TODO
        except:
            print("=> %s does not exist!", self.initial_weight)
            print("=> Now it starts to train YOLOv3 from scratch ...")
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                pass    # TODO
            else:
                pass    # TODO

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            # batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
            # batch_sbboxes, batch_mbboxes, batch_lbboxes
            for train_data in pbar:
                train_step_loss, global_step_val = self.sess.run(
                    [self.loss, self.global_step], feed_dict={
                        self.input_data: train_data[0],
                        self.label_sbbox: train_data[1],
                        self.label_mbbox: train_data[2],
                        self.label_lbbox: train_data[3],
                        self.true_sbboxes: train_data[4],
                        self.true_mbboxes: train_data[5],
                        self.true_lbboxes: train_data[6],
                        self.trainable: True,
                    })

                print(train_epoch_loss, global_step_val)
            break
        pass


    def _test(self):
        print(self.trainset)


if __name__ == '__main__':
    # YoloTrain().train()
    YoloTrain()._test()
