# -*- coding: utf-8 -*-
import copy
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageProcessing(object):
    last_death_check = True
    last_last_death_check = True
    last_last_last_death_check = True

    death_check_log_data_size = 3
    death_check_log = [True]*death_check_log_data_size
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

        self.roi_mask = self.to_grayscale(cv2.imread("./ROI1.png"))
        #self.roi_mask = self.to_grayscale(cv2.imread("./ROI2.png"))
        print "roi_mask[150,0]="
        print self.roi_mask[150,0]

    def to_plot(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def to_grayscale(self,img):
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return grayed

    # reshape済みの配列が引数,グレースケールのimgをreturn
    def lane_detection(self,img):
        img_gray = self.to_grayscale(img)

        #kernel = np.ones((5,5),np.float32)/25
        #img_gray = cv2.filter2D(img_gray,-1,kernel)
        #mask = cv2.threshold(img_gray,180,255,cv2.THRESH_BINARY)[1]

        mask = cv2.inRange(img_gray, 200, 255)
        return mask

    def check_death(self,gray_img,revived):
        if(revived):
            alive_value = 1600
            #return np.sum(gray_img[:113,:]/255.0)<alive_value
        else:
            alive_value = 800
        death_check = np.sum(gray_img[:140,:]/255.0)<alive_value
        self.death_check_log.append(death_check)
        self.death_check_log.pop(0)
        print self.death_check_log
        return_death = (np.sum(self.death_check_log)==self.death_check_log_data_size)
        return return_death


    def change_speed(self,gray_img):
        hoge1 = np.sum(self.roi_mask[:,:113] * gray_img[:,:113])
        hoge2 = np.sum(self.roi_mask[:,113:] * gray_img[:,113:])
        if(hoge1==0 or hoge2==0):
            hoge3 = 0
        elif(hoge1<hoge2):
            hoge3 =  float(hoge1)/float(hoge2)
        elif(hoge1>hoge2):
            hoge3 =  float(hoge2)/float(hoge1)
        elif(hoge1 == hoge2):
            hoge3 = 1
        return hoge1,hoge2,hoge3


    def make_detection_image(self,src,mask):
        flip_mask = 255-mask
        src_copy = src.copy()
        src_copy[:,:,0] = cv2.bitwise_and(src_copy[:,:,0],src_copy[:,:,0], mask=flip_mask)
        src_copy[:,:,1] = cv2.bitwise_and(src_copy[:,:,1],src_copy[:,:,1], mask=flip_mask)
        return src_copy

    def save_image(self,img,photo_id,dir_path='./SaveImage/'):
        img_name = '%05d.png'%(photo_id)
        cv2.imwrite(dir_path+img_name,img)
        print 'Save Finish : %s'%(dir_path+img_name)

    def plot(self,img1,img2,q,title='',a_num=3):
        if(self.plot_image_num == 1):
            self.ax1.cla()
            self.ax1.title.set_text(title)
            self.ax1.imshow(img1)
            if(self.plot_q_value):
                self.plot_q(q,a_num)
            plt.pause(1.0 / 10**10)

        elif(self.plot_image_num == 2):
            self.ax1.cla()
            self.ax2.cla()
            self.ax1.title.set_text(title)
            self.ax1.imshow(img1)
            self.ax2.imshow(img2)
            if(self.plot_q_value):
                self.plot_q(q,a_num)
            plt.pause(1.0 / 10**10)

    def plot_q(self,q,a_num=3):
        self.ax3.cla()
        actions = range(a_num)
        max_q_abs = max(abs(q))
        if max_q_abs != 0:
            q = q / float(max_q_abs)
        self.ax3.set_xticks(actions)
        if(a_num==3):
            self.ax3.set_xticklabels(['left','forward','right'], rotation=0, fontsize='small')
        elif(a_num==5):
            self.ax3.set_xticklabels(['left','forward','right','accel','brake'], rotation=0, fontsize='small')
        self.ax3.set_xlabel("Action") # x軸のラベル
        self.ax3.set_ylabel("Q_Value") # y軸のラベル
        self.ax3.set_ylim(-1.1, 1.1)  # yを-1.1-1.1の範囲に限定
        self.ax3.set_xlim(-1, a_num) # xを-0.5-7.5の範囲に限定
        self.ax3.hlines(y=0, xmin=-1, xmax=a_num, colors='r', linewidths=2) #y=0の直線
        self.ax3.bar(actions,q,align="center")

    def main_check(self):
        imgDir_path = './RoboCarImage/'
        #imgDir_path = './SaveImage/'
        #imgDir_path = './SaveImageStraight/'

        file_list = os.listdir(imgDir_path)
        for File in file_list:
            if '.jpg' not in File and '.jpeg' not in File and '.png' not in File and '.JPG' not in File:
                file_list.remove(File)

        sum_image = np.zeros((227,227))
        for i in range(len(file_list)):
            a_num = 5
            q = np.random.rand(a_num)
            image = cv2.imread(imgDir_path+file_list[i])
            image = cv2.resize(image,(227,227))
            new_image_g = self.lane_detection(image)
            new_image = cv2.merge((new_image_g,new_image_g,new_image_g))
            hoge = np.sum(new_image_g/255.0)

            #hoge = np.sum(self.roi_mask * new_image_g)
            #self.save_image(new_image_g,photo_id=hoge)
            #cv2.imwrite("./SaveImage/%d,%d.png"%(hoge1,hoge2),new_image_g)
            #self.save_image(new_image_g,photo_id=hoge,dir_path='SaveImage2/')

            #sum_image += new_image_g
            self.plot(new_image_g,new_image_g,q.ravel(),title=hoge,a_num=a_num)
            #self.plot(sum_image,sum_image,q.ravel(),a_num=a_num)
            #self.plot(self.to_plot(image),new_image,q.ravel(),a_num=a_num)
            #self.plot(self.to_plot(image),self.to_plot(self.make_detection_image(image,new_image_g)),q.ravel(),title=title,,a_num=a_num)
        #self.save_image(sum_image,photo_id=0)





if __name__ == '__main__':
    import time
    start_time = time.time()
    img_pro = ImageProcessing(1,False)
    img_pro.main_check()
    run_time = time.time()-start_time
    print "Run Time : %.3f"%(run_time)
