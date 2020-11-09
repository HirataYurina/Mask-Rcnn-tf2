# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:roi_align.py
# software: PyCharm


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class RoiAlign(layers.Layer):

    def __init__(self):
        super(RoiAlign, self).__init__()

    def call(self, inputs, **kwargs):
        pass