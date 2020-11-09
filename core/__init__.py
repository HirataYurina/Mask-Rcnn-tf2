# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:__init__.py.py
# software: PyCharm

import tensorflow as tf
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(tf.nn.top_k(a, k=1))