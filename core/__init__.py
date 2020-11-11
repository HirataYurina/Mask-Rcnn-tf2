# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:__init__.py.py
# software: PyCharm

import tensorflow as tf
import numpy as np

a = tf.random.normal(shape=(10, 4))
b = tf.split(a, 4, axis=1)
print(b)

print(tf.maximum(tf.math.log(0.0) + 4, 2))

c = tf.constant([[0, 1, 10, 3, 6],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2]])
print(tf.nn.top_k(c, 4))

d = tf.constant([[0], [1]])

print('tf.gather', tf.gather(c, d))
print(tf.where(c > 3))
print('tf.where', tf.gather_nd(c, tf.where(c > 3)))