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
    rpn_class_logits = layers.Conv2D(2 * num_anchors, 1, name='rpn_class_raw')(shared_feature)
    rpn_class_logits = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(rpn_class_logits)
    rpn_class_probs = layers.Softmax(name='rpn_class_xxx')(rpn_class_logits)

    # the location
    # [batch_size, none, none, 4 * num_anchors]
    location = layers.Conv2D(4 * num_anchors, 1, name='rpn_bbox_pred')(shared_feature)
    location = layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(location)  # [batch, num_anchors, 4]

    return rpn_class_logits, rpn_class_probs, location


def refine_network_header(roi_feature, bn_train, num_classes):
    """classifier and location head of fpn"""

    # TODO: add name to layers
    # roi_feature: [batch, num_rois, 7, 7, channels]
    x = layers.TimeDistributed(layers.Conv2D(1024, 7, padding='valid'), name='mrcnn_class_conv1')(roi_feature)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='mrcnn_class_bn1')(x, bn_train)
    x = layers.ReLU()(x)  # [batch, num_rois, 1, 1, 1024]

    x = layers.TimeDistributed(layers.Conv2D(1024, 1, padding='valid'), name='mrcnn_class_conv2')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='mrcnn_class_bn2')(x, bn_train)
    x = layers.ReLU()(x)  # [batch, num_rois, 1, 1, 1024]

    shared = layers.Lambda(lambda t: tf.squeeze(tf.squeeze(t, axis=3), axis=2))(x)

    # classification
    classi_logits = layers.TimeDistributed(layers.Dense(num_classes), name='mrcnn_class_logits')(shared)
    classification = layers.TimeDistributed(layers.Softmax(), name='mrcnn_class')(classi_logits)

    # bounding box
    bbox = layers.TimeDistributed(layers.Dense(4 * num_classes), name='mrcnn_bbox_fc')(shared)
    bbox = layers.Reshape(target_shape=(-1, num_classes, 4), name='mrcnn_bbox')(bbox)

    return classi_logits, classification, bbox


def mask_header(roi_feature, bn_train, num_classes):
    """mask head of Feature Pyramid Network"""

    # TODO: add name to layers
    x = layers.TimeDistributed(layers.Conv2D(256, 3, padding='same'), name='mrcnn_mask_conv1')(roi_feature)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='mrcnn_mask_bn1')(x, training=bn_train)
    x = layers.ReLU()(x)

    x = layers.TimeDistributed(layers.Conv2D(256, 3, padding='same'), name='mrcnn_mask_conv2')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='mrcnn_mask_bn2')(x, training=bn_train)
    x = layers.ReLU()(x)

    x = layers.TimeDistributed(layers.Conv2D(256, 3, padding='same'), name='mrcnn_mask_conv3')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='mrcnn_mask_bn3')(x, training=bn_train)
    x = layers.ReLU()(x)

    x = layers.TimeDistributed(layers.Conv2D(256, 3, padding='same'), name='mrcnn_mask_conv4')(x)
    x = layers.TimeDistributed(layers.BatchNormalization(), name='mrcnn_mask_bn4')(x, training=bn_train)
    x = layers.ReLU()(x)

    x = layers.TimeDistributed(layers.Conv2DTranspose(256, 2, strides=2), name='mrcnn_mask_deconv')(x)
    x = layers.ReLU()(x)
    # compute inference of mask
    # [batch, num_roi, height, width, num_classes]
    x = layers.TimeDistributed(layers.Conv2D(num_classes, 1, activation='sigmoid'),
                               name='mrcnn_mask')(x)

    return x
