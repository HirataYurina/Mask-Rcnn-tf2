# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:fpn.py
# software: PyCharm

from net.resnet import ResNet
import tensorflow.keras.layers as layers
import tensorflow.keras as keras


def neck_network(c2, c3, c4, c5):
    p5 = layers.Conv2D(256, 1, name='fpn_c5p5')(c5)

    p5_up = layers.UpSampling2D(name='fpn_p5upsampled')(p5)
    p4 = layers.Conv2D(256, 1, name='fpn_c4p4')(c4)
    p4 = layers.Add(name='fpn_p4add')([p5_up, p4])

    p3 = layers.Conv2D(256, 1, name='fpn_c3p3')(c3)
    p4_up = layers.UpSampling2D(name='fpn_p4upsampled')(p4)
    p3 = layers.Add(name='fpn_p3add')([p4_up, p3])

    p3_up = layers.UpSampling2D(name='fpn_p3upsampled')(p3)
    p2 = layers.Conv2D(256, 1, name='fpn_c2p2')(c2)
    p2 = layers.Add(name='fpn_p2add')([p3_up, p2])

    # Attach 3x3 conv to all P layers to get the final feature maps.
    p2 = layers.Conv2D(256, kernel_size=3, padding='same', name='fpn_p2')(p2)
    p3 = layers.Conv2D(256, kernel_size=3, padding='same', name='fpn_p3')(p3)
    p4 = layers.Conv2D(256, kernel_size=3, padding='same', name='fpn_p4')(p4)
    p5 = layers.Conv2D(256, kernel_size=3, padding='same', name='fpn_p5')(p5)
    # p6 is generated by subsampling from p5 with stride of 2
    p6 = layers.MaxPool2D(pool_size=1, strides=2, name='fpn_p6')(p5)

    return p2, p3, p4, p5, p6


if __name__ == '__main__':
    resnet_50 = ResNet(50)
    inputs_ = keras.Input(shape=(608, 608, 3))
    c2_, c3_, c4_, c5_ = resnet_50(inputs_)
    p2_, p3_, p4_, p5_, p6_ = neck_network(c2_, c3_, c4_, c5_)

    # Total
    # params: 26, 905, 536
    # Trainable
    # params: 26, 852, 416
    # Non - trainable
    # params: 53, 120
    model = keras.Model(inputs_, [p2_, p3_, p4_, p5_, p6_])
    model.summary()
