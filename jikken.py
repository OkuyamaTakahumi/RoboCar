# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np

import matplotlib.pyplot as plt

from chainer import cuda



print 1

range(9)+1
a = (np.array(range(9),dtype = np.uint8)+1).reshape((9,1))

a
a2=a.ravel()
a2

a[0]
a2[0]

a.ravel()[0]
c = a.astype(np.float32)
c
b = np.frombuffer(a.data, dtype=np.uint);
b = True
not b
