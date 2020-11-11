# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:config.py
# software: PyCharm


import easydict

a = easydict.EasyDict()
a.techi = 1
a.num_classes = 1 + 80  # add background to num_classs

print(a.num_classes)