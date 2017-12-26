# coding: utf-8

import numpy as np
from chainer import optimizers, Variable, serializers, Function, cuda
import cv2
import os
from FCN import FCN
from PIL import Image
import time


class LaneDetection:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        LOAD_MODEL_NAME = "1lane.npz"
        #LOAD_MODEL_NAME = "2lane.npz"

        # original_segmentations = Image.open("./palette.png")
        # palette = original_segmentations.getpalette()

        # モデルの設定
        self.model = FCN()
        serializers.load_npz("./AdatiDir/%s"%(LOAD_MODEL_NAME), self.model)

        optimizer = optimizers.Adam()
        optimizer.setup(self.model)

        cuda.get_device(0).use()
        self.model.to_gpu()

    def mainfunction(self, img):
        # start = time.time()

        # print "start"

        img_resize = cv2.resize(img, (224, 224))
        img_resize = img_resize.transpose(2, 0, 1)

        train_img = np.array([img_resize],dtype = np.float32)

        x = Variable(cuda.to_gpu(train_img))
        self.model.zerograds()
        self.model.train = False
        loss = self.model(x, 1)
        cpu = cuda.to_cpu(loss.data)

        imgarray = cpu[0, 1, :, :]
        imgarray[:, :] = np.round(imgarray[:, :])  # 四捨五入

        #imgarray = imgarray * 255
        imgarray = np.array(imgarray*255,dtype = np.uint8)

        # img = Image.fromarray(np.uint8(imgarray))
        return_img = cv2.merge((imgarray, imgarray, imgarray))
        return return_img

# ------------------------------------------------------------------------------------

# start = time.time()
# img = cv2.imread("./00000_0.000000.png")
# img = img.transpose(2, 0, 1).astype(np.float32)
#
# hoge = LaneDetection()
# img = hoge.mainfunction(img, 0)
#
# # new_img = cv2.merge((img, img, img))
# # print img.shape
#
# img2 = np.frombuffer(img,dtype = "uint8")
# print img2.shape
