# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:head.py
# software: PyCharm

import tensorflow.keras.layers as layers
import tensorflow as tf


def rpn_header(feature, num_anchors):
    """ the header of region proposal network
                                                                                    / scores_branch
    x -> resnet -> c2, c3, c4, c5 -> FPN -> p2, p3, p4, p5, p6 --> shared_layer -->
                                                                                    \ location_branch
    Args:
        feature:     the outputs of FPN
        num_anchors: the channels is 2 * num_anchors and 4 * num_anchors

    Returns:
        rpn_class_logits
        rpn_class_probs
        location

    """
    shared_feature = layers.Conv2D(512, 3, padding='same', name='rpn_conv_shared')(feature)
    shared_feature = layers.ReLU()(shared_feature)

    # scores of positive negative
    rpn_class_logits = layers.Conv2D(2 * num_anchors, name='rpn_class_raw')(shared_feature)
    rpn_class_logits = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(rpn_class_logits)
    rpn_class_probs = layers.Softmax(name='rpn_class_xxx')(rpn_class_logits)

    # the location
    # [batch_size, none, none, 4 * num_anchors]
    location = layers.Conv2D(4 * num_anchors, name='rpn_bbox_pred')(shared_feature)
    location = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(location)  # [num_anchors, 4]

    return rpn_class_logits, rpn_class_probs, location
