# coding: utf-8

import numpy as np
import chainer
from chainer import optimizers, Variable, serializers, Function, cuda
import chainer.links as L
import chainer.functions as F
import math
from add import add
import cupy

class FCN(chainer.Chain):
    CLASSES = 2
    IN_SIZE = 224

    def __init__(self):
        super(FCN, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            score_pool3=L.Convolution2D(256, FCN.CLASSES, 1, stride=1, pad=0),
            score_pool4=L.Convolution2D(512, FCN.CLASSES, 1, stride=1, pad=0),
            score_pool5=L.Convolution2D(512, FCN.CLASSES, 1, stride=1, pad=0),

            upsample_pool4=L.Deconvolution2D(FCN.CLASSES, FCN.CLASSES, ksize=4, stride=2, pad=1),
            upsample_pool5=L.Deconvolution2D(FCN.CLASSES, FCN.CLASSES, ksize=8, stride=4, pad=2),
            upsample_final=L.Deconvolution2D(FCN.CLASSES, FCN.CLASSES, ksize=16, stride=8, pad=4),

            norm3=L.BatchNormalization(256),
            norm4=L.BatchNormalization(512),
            norm5=L.BatchNormalization(512),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = self.norm3(h, test=not self.train)
        h = F.max_pooling_2d(h, 2, stride=2)
        pool3 = h

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = self.norm4(h, test=not self.train)
        h = F.max_pooling_2d(h, 2, stride=2)
        pool4 = h

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = self.norm5(h, test=not self.train)
        h = F.max_pooling_2d(h, 2, stride=2)
        pool5 = h

        p3 = self.score_pool3(pool3)
        self.p3_shape = p3.data.shape

        p4 = self.score_pool4(pool4)
        self.p4_shape = p4.data.shape

        p5 = self.score_pool5(pool5)
        self.p5_shape = p5.data.shape

        u4 = self.upsample_pool4(p4)
        self.u4_shape = u4.data.shape

        u5 = self.upsample_pool5(p5)
        self.u5_shape = u5.data.shape

        h = add(p3, u4, u5)

        h = self.upsample_final(h)
        self.final_shape = h.data.shape

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            if math.isnan(self.loss.data):
                raise RuntimeError("ERROR in Fcn: loss.data is nan!")
            return self.loss,self.accuracy
        else:
            self.pred = F.softmax(h)
            return self.pred