#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Time    : 10/21/20 2:24 PM
# Author  : Shark
# Site    : 
# File    : utils.py
# Software: PyCharm
# ===========================================================
import numpy as np


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


if __name__ == "__main__":
    _names = read_class_names("../data/classes/voc.names")
    print(_names)

    _anchors = get_anchors("../data/anchors/baseline_anchors.txt")
    print(_anchors)
