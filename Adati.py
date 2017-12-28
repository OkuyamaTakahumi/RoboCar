# coding: utf-8

import numpy as np
from chainer import optimizers, Variable, serializers, Function, cuda
import cv2
import os
from AdatiDir.FCN import FCN
from PIL import Image
import time

from image_processing import ImageProcessing

class ImageProcessing_adati(ImageProcessing):
    def __init__(self,plot_image_num,plot_q_value):
        super(ImageProcessing_adati, self).__init__(plot_image_num,plot_q_value)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        #LOAD_MODEL_NAME = "1lane.npz"
        LOAD_MODEL_NAME = "2lane.npz"

        # original_segmentations = Image.open("./palette.png")
        # palette = original_segmentations.getpalette()

        # モデルの設定
        self.model = FCN()
        serializers.load_npz("./AdatiDir/%s"%(LOAD_MODEL_NAME), self.model)

        optimizer = optimizers.Adam()
        optimizer.setup(self.model)

        cuda.get_device().use()
        self.model.to_gpu()

    def lane_detection(self, img):
        # start = time.time()

        print "Adati Detection"

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

        imgarray = np.array(imgarray*255,dtype = np.uint8)
        imgarray = cv2.resize(imgarray,(227,227))
        return imgarray

if __name__ == '__main__':
    import time
    start_time = time.time()
    img_pro = ImageProcessing_adati(2,False)
    img_pro.main_check()
    run_time = time.time()-start_time
    print "Run Time : %.3f"%(run_time)
