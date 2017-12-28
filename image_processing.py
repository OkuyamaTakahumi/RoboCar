# -*- coding: utf-8 -*-
import copy
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageProcessing(object):
    def __init__(self,plot_image_num,plot_q_value):
        self.plot_image_num = plot_image_num
        self.plot_q_value = plot_q_value
        if(plot_image_num==2 and plot_q_value):
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3)
            plt.gray()
        elif(plot_image_num==1 and plot_q_value):
            self.fig, (self.ax1, self.ax3) = plt.subplots(1, 2)
            plt.gray()
        elif(plot_image_num==2):
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
            plt.gray()
        elif(plot_image_num==1):
            self.fig, self.ax1 = plt.subplots(1, 1)
            plt.gray()
        elif(plot_q_value):
            self.fig, self.ax3 = plt.subplots(1, 1)

    def to_plot(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def to_grayscale(self,img):
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayed

    # reshape済みの配列が引数,グレースケールのimgをreturn
    def lane_detection(self,img):
        img_gray = self.to_grayscale(img)

        #kernel = np.ones((5,5),np.float32)/25
        #img_resize = cv2.filter2D(img_gray,-1,kernel)
        #mask = cv2.threshold(img_resize,180,255,cv2.THRESH_BINARY)[1]

        mask = cv2.inRange(img_gray, 200, 255)
        return mask

    def make_detection_image(self,src,mask):
        flip_mask = 255-mask
        src_copy = src.copy()
        src_copy[:,:,0] = cv2.bitwise_and(src_copy[:,:,0],src_copy[:,:,0], mask=flip_mask)
        #src_copy[:,:,1] = cv2.bitwise_and(src_copy[:,:,1],src_copy[:,:,1], mask=flip_mask)
        return src_copy

    def save_image(self,img,photo_id):
        img_name = '%d.png'%(photo_id)
        cv2.imwrite('./SaveImage/'+img_name,img)
        print 'Save Finish : %s'%(img_name)

    def plot(self,img1,img2,q,a_num=13):
        if(self.plot_image_num == 1):
            self.ax1.cla()
            #self.ax1.tick_params(labelleft="off",labelbottom='off')
            #self.ax1.title.set_text('Lane Detection Image')
            self.ax1.imshow(img1)
            if(self.plot_q_value):
                self.plot_q(q,a_num)
            plt.pause(1.0 / 10**10)
        elif(self.plot_image_num == 2):
            self.ax1.cla()
            self.ax2.cla()
            self.ax1.imshow(img1)
            self.ax2.imshow(img2)
            if(self.plot_q_value):
                self.plot_q(q,a_num)
            plt.pause(1.0 / 10**10)

    def plot_q(self,q,a_num):
        self.ax3.cla()
        actions = range(a_num)
        max_q_abs = max(abs(q))
        if max_q_abs != 0:
            q = q / float(max_q_abs)
        self.ax3.set_xticks(actions)
        if(a_num==7):
            self.ax3.set_xticklabels(['-30','-20','-10','0','10','20','30'], rotation=0, fontsize='small')
        elif(a_num==13):
            self.ax3.set_xticklabels(['-30','-25','-20','-15','-10','-5','0','5','10','15','20','25','30'], rotation=0, fontsize='small')
        self.ax3.set_xlabel("Action") # x軸のラベル
        self.ax3.set_ylabel("Q_Value") # y軸のラベル
        self.ax3.set_ylim(-1.1, 1.1)  # yを-1.1-1.1の範囲に限定
        self.ax3.set_xlim(-1, a_num) # xを-0.5-7.5の範囲に限定
        self.ax3.hlines(y=0, xmin=-1, xmax=a_num, colors='r', linewidths=2) #y=0の直線
        self.ax3.bar(actions,q,align="center")

    def main_check(self):
        imgDir_path = './RoboCarImage/'

        file_list = os.listdir(imgDir_path)
        for File in file_list:
            if '.jpg' not in File and '.jpeg' not in File and '.png' not in File and '.JPG' not in File:
                file_list.remove(File)

        for i in range(len(file_list)):
            image = cv2.imread(imgDir_path+file_list[i])
            image = cv2.resize(image,(227,227))
            new_image_g = self.lane_detection(image)
            new_image = cv2.merge((new_image_g,new_image_g,new_image_g))
            #self.save_image(new_image,i)
            q = np.random.rand(13)
            self.plot(self.to_plot(image),new_image,q.ravel())
            #self.plot(self.to_plot(image),self.make_detection_image(image,new_image_g),q.ravel())

if __name__ == '__main__':
    import time
    start_time = time.time()
    img_pro = ImageProcessing(2,False)
    img_pro.main_check()
    run_time = time.time()-start_time
    print "Run Time : %.3f"%(run_time)
