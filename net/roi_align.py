# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:roi_align.py
# software: PyCharm


import tensorflow as tf
# import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from util.utils import parse_image_meta_graph


class RoiAlign(layers.Layer):
    """implement ROI pooling on multiple levels of the feature pyramid.

    inputs:
        boxes:        [batch_size, num_boxes, (y1, x1, y2, x2)] in normalized coordinates.
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
        image_shape = parse_image_meta_graph(images_meta)['image_shape']
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)

        # compute proposals area
        # TODO: use tf.split(boxes, 4, axis=2)
        y1 = boxes[..., 0]
        x1 = boxes[..., 1]
        y2 = boxes[..., 2]
        x2 = boxes[..., 3]
        w = x2 - x1  # [batch_size, num_boxes]
        h = y2 - y1  # [batch_size, num_boxes]

        # use this equation to compute proposals belong to which level
        # this equation can be referenced in "Feature Pyramid Networks for Object Detection"
        # the w and h in normalized, so we need to use w * image_w to get the w in image coordinate
        # if we have an ROI with w*h=224*224, the level will be 4(C4)
        # TODO: tf.math.log can not feed int data
        # TODO: tf.math.log(0) = -inf and tf.maximum(-inf, 2) = 2
        levels_equation = tf.math.log(tf.sqrt(w * h) * tf.sqrt(image_area) / 224.0) / tf.math.log(2.0)
        levels_equation = tf.cast(tf.round(levels_equation), tf.int32)
        # crop the levels
        # and if the boxes are padded by zero, the w and h is zero too
        # so, the areas of these proposals are zero
        levels_cropped = tf.maximum(2, tf.minimum(4 + levels_equation, 5))  # [batch_size, num_boxes]

        # use pool size and proposals to crop and resize feature maps
        feature_pooled = []
        feature_box_id = []
        for i, level in enumerate(tf.range(2, 6)):
            # i is used as index of feature list
            feature_level = feature_maps[i]
            # select ids that equal to level
            id_level = tf.where(tf.equal(levels_cropped, level))  # [num_boxes, 2]
            boxes_to_level = tf.gather_nd(boxes, id_level)
            id_to_batch = id_level[..., 0]
            # [num_boxes, pool_size, pool_size, depth]
            feature_pooled_level = tf.image.crop_and_resize(image=feature_level,
                                                            boxes=boxes_to_level,
                                                            box_indices=id_to_batch,
                                                            crop_size=self.pool_size)
            feature_pooled.append(feature_pooled_level)
            feature_box_id.append(id_level)

        # pack pooled feature into one tensor
        # pack feature box id into one tensor
        feature_pooled = tf.concat(feature_pooled, axis=0)  # [num_boxes, pool_size, pool_size, depth]
        feature_box_id = tf.concat(feature_box_id, axis=0)  # [num_boxes, (batch, box_id)]
        # sort feature_box_id by batch and box_id
        sort_tensor = 100000 * feature_box_id[..., 0] + feature_box_id[..., 1]
        id_sorted = tf.nn.top_k(sort_tensor, tf.shape(feature_box_id)[0]).indices[::-1]

        feature_pooled_sort = tf.gather(feature_pooled, id_sorted)

        # re-add batch and num_boxes dimension
        feature_pooled_sort = tf.reshape(feature_pooled_sort,
                                         shape=(tf.concat(tf.shape(boxes)[:2], tf.shape(feature_pooled)[1:])))

        return feature_pooled_sort
