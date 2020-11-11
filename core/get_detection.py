# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:get_detection.py
# software: PyCharm

import tensorflow as tf
# import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from util.utils import delta2box


class ProposalLayer(layers.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage.
    And the nms_limit=6000
            post_nms_roi_training=2000
            post_nms_roi_inference=1000

    inputs:
        rpn_probs: [batch_size, num_anchors, (negative, positive)]
        rpn_bbox:  [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors:   [batch_size, num_anchors, (y1, x1, y2, x2)]

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """
    
    def __init__(self,
                 proposal_count,
                 # score_thres,
                 config,
                 rpn_bbox_std_dev=tf.constant([0.1, 0.1, 0.2, 0.2]),
                 **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        # self.score_thres = score_thres
        self.config = config
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
    
    def call(self, inputs, **kwargs):
        # scores
        positive_score = inputs[0][..., 1]  # [batch_size, num_anchors]
        # location delta
        delta = inputs[1]  # [batch, num_anchors, 4]
        delta = delta * tf.reshape(self.rpn_bbox_std_dev, shape=(1, 1, 4))
        # anchors
        anchors = inputs[2]
        # batch_size
        batch_size = tf.shape(positive_score)[0]
        num_anchors = tf.shape(positive_score)[1]

        total_proposals = []

        for i in range(batch_size):
            batch_score = positive_score[i]  # [num_anchors,]
            batch_delta = delta[i]  # [num_anchors, 4]
            batch_anchors = anchors[i]  # [num_anchors, 4]
            # 1.use score threshold to get positive targets [×]
            # this is different from detection layer of one stage detector because one stage detector
            # uses score threshold to select positive targets
            # -------------------------------------------------
            # 2.use anchors and delta to decode bounding boxes
            boxes = delta2box(batch_delta, batch_anchors)
            nms_limit = tf.minimum(self.config.nms_limit, num_anchors)
            id_limit = tf.nn.top_k(batch_score, k=nms_limit)
            score_sorted = tf.gather(batch_score, id_limit)
            boxes_sorted = tf.gather(boxes, id_limit)

            # 3.crop the bounding boxes

            # 4.use nms to get final detections
            picked = tf.image.non_max_suppression(boxes=boxes_sorted,
                                                  scores=score_sorted,
                                                  max_output_size=self.proposal_count,
                                                  name='rpn_non_max_suppression')

            # 5.use self.proposal_count to get proposal targets [√]
            # and pad proposal matrix with zero to avoid shape mismatch
            # so, we can get [batch, rois, (y1, x1, y2, x2)]
            proposals = tf.gather(boxes_sorted, picked)  # [num_picked, 4] in image coordinate
            pad_count = self.proposal_count - tf.shape(proposals)[0]
            proposals_padded = tf.pad(proposals,
                                      paddings=tf.constant([[0, pad_count], [0, 0]]))
            total_proposals.append(proposals_padded)

        # change list to tensor
        total_proposals = tf.stack(total_proposals, axis=0)

        return total_proposals


class DetectionLayer(layers.Layer):
    """The detection layer that used to process results inferred by second stage

    inputs:
        classification: [batch, num_rois, num_classes(have background)]
        bbox:           [batch, num_rois, num_classes, 4]
        image_meta:     [batch, image_meta]
        rois:           [batch, num_rois, 4]

    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs, **kwargs):
        classification = inputs[0]
        bbox = inputs[1]
        image_meta = inputs[2]
        roi = inputs[4]

        batch_size = tf.shape(classification)[0]

        scores_batch = []
        boxes_batch = []
        classes_batch = []
        for i in range(batch_size):
            per_classification = classification[i]
            per_bbox = bbox[i]
            # TODO: get window
            per_image_meta = image_meta[i]
            per_roi = roi[i]
            # 1.select max id in classification [num_boxes, num_classes(coco:1(background)+80(classes))]
            max_ids = tf.argmax(per_classification, axis=1, output_type=tf.int32)  # [num_rois,]
            num_boxes = tf.shape(per_classification)[0]
            indices = tf.stack([tf.range(num_boxes), max_ids], axis=1)

            scores = tf.gather_nd(per_classification, indices)  # [num_rois,]
            boxes = tf.gather_nd(per_bbox, indices)  # [num_rois, 4]

            # 2.select positive targets
            positive_targets = tf.where(max_ids > 0)[..., 0]  # [num_positives,]
            positive_indices = tf.gather(indices, positive_targets)  # [num_positives, 2]
            pos_scores = tf.gather(scores, positive_targets)  # [num_positives,]
            pos_boxes = tf.gather(boxes, positive_targets)  # [num_positives, 4]
            pos_rois = tf.gather(per_roi, positive_targets)  # [num_positives, 4]

            # 3.delta to box
            # TODO: write delta2box function
            pos_boxes = delta2box(pos_boxes, pos_rois)

            # 4.crop box
            # TODO: use window to crop box

            scores = []
            boxes = []
            classes = []
            # 5.apply nms to every class
            for class_id in range(1, 81):
                pos_to_class = tf.where(tf.equal(positive_indices[..., 1], class_id))[..., 0]  # [num_class,]
                scores_class = tf.gather(pos_scores, pos_to_class)  # [num_class,]
                boxes_class = tf.gather(pos_boxes, pos_to_class)  # [num_class, 4]
                pick = tf.image.non_max_suppression(boxes=boxes_class,
                                                    scores=scores_class,
                                                    max_output_size=100,
                                                    iou_threshold=self.config.refine_iou_threshold,
                                                    score_threshold=self.config.refine_score_threshold)
                scores_ = tf.gather(scores_class, pick)
                boxes_ = tf.gather(boxes_class, pick)
                classes_ = tf.ones_like(scores_, dtype=tf.int32) * class_id
                scores.append(scores_)
                boxes.append(boxes_)
                classes.append(classes_)

            scores_batch.append(scores)
            boxes_batch.append(boxes)
            classes_batch.append(classes)

        return scores_batch, boxes_batch, classes_batch


def mask_detection():
    # 1.select mask with threshold(0.5 is recommended in "mask rcnn" paper)

    # 2.resize 28*28 to proposals' scale

    # 3.put masks into original image
    pass


def detection():
    # 1.decode the proposals in model coordinate to original coordinate

    # 2.filter out mask by id of proposals

    # 3.put mask into every image

    pass
