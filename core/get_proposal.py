# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:get_proposal.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np


class ProposalLayer(layers.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage

    inputs:
        rpn_probs: [batch_size, num_anchors, (negative, positive)]
        rpn_bbox:  [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors:   [batch_size, num_anchors, (y1, x1, y2, x2)]

    """
    
    def __init__(self,
                 proposal_count,
                 score_thres,
                 config,
                 rpn_bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
                 **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.score_thres = score_thres
        self.config = config
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
    
    def call(self, inputs, **kwargs):
        # scores
        positive_score = inputs[0][..., 1]  # [batch_size, num_anchors]
        # location delta
        delta = inputs[1]  # [batch, num_anchors, 4]
        delta = delta * np.reshape(self.rpn_bbox_std_dev, newshape=(1, 1, 4))
        # anchors
        anchors = inputs[2]







