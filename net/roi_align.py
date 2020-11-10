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
    """implement ROI pooling on multiple levels of the feature pyramid.

    inputs:
        boxes:        [batch_size, num_boxes, (y1, x1, y2, x2)]
                      Possibly padded with zero if not enough boxes to fill this array.
        image_meta:   [batch_size, meta_data]
        feature_maps: [batch_size, height, width, channels]
                      the list of different level feature map
    """

    def __init__(self, pool_size):
        super(RoiAlign, self).__init__()
        self.pool_size = pool_size

    def call(self, inputs, **kwargs):
        boxes = inputs[0]
        images_meta = inputs[1]
        feature_maps = inputs[2]

        # compute proposals area
        # TODO: use tf.split(boxes, 4, axis=2)
        y1 = boxes[..., 0]
        x1 = boxes[..., 1]
        y2 = boxes[..., 2]
        x2 = boxes[..., 3]
        w = x2 - x1  # [batch_size, num_boxes]
        h = y2 - y1  # [batch_size, num_boxes]

        # use this equation to compute proposals belong to which level
        levels_equation = tf.math.log(tf.sqrt(w * h) * tf.sqrt(image_area) / 224.0) / tf.math.log(2)
        # crop the levels
        levels_cropped = tf.maximum(2, tf.minimum(levels_equation, 5))

        # use pool size and proposals to crop and resize feature maps

