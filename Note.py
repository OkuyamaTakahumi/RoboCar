# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import numpy as np
import os
print 3
plt.gray()

def make_detection_image(img,mask):
    flip_mask = 255-mask
    img[:,:,0] = cv2.bitwise_and(img[:,:,0],img[:,:,0], mask=flip_mask)
    img[:,:,1] = cv2.bitwise_and(img[:,:,1],img[:,:,1], mask=flip_mask)
    return img

def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ------------------------------------ROI--------------------------------------------------
image1 = cv2.imread('C:\Users\Student2015\Desktop\RoboCar\straight1.png')
image2 = cv2.imread('C:\Users\Student2015\Desktop\RoboCar\straight2.png')
plt.imshow(image1)
plt.imshow(image2)
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
plt.gray()

ROI1 = gray_image1+cv2.flip(gray_image1,1)
ROI1 = cv2.rectangle(ROI1, (0, 0), (227, 50), (0, 0, 0), -1)
ROI2 = gray_image2+cv2.flip(gray_image2,1)
ROI2 = cv2.rectangle(ROI2, (0, 0), (227, 16), (0, 0, 0), -1)
plt.imshow(ROI1)
plt.imshow(ROI2)
cv2.imwrite("ROI1.png",ROI1)
cv2.imwrite("ROI2.png",ROI2)

ROI1[150,0]
ROI2[150,0]
# -------------------------------Image ImageProcessing------------------------------------------------

imgDir_path = '/Users/okuyamatakashi/RoboCar/RoboCarImage/'
imgDir_path = 'C:\Users\Student2015\Desktop/RoboCar/RoboCarImage/'

file_list = os.listdir(imgDir_path)
for File in file_list:
    if '.jpg' not in File and '.jpeg' not in File and '.png' not in File and '.JPG' not in File:
        file_list.remove(File)

image = cv2.imread(imgDir_path+file_list[224])
image1 = cv2.imread(imgDir_path+file_list[90])
image2 = cv2.imread(imgDir_path+file_list[20])
image = cv2.resize(image,(227,227))
image1 = cv2.resize(image1,(227,227))
image2 = cv2.resize(image2,(227,227))
plt.imshow(to_plot(image))
plt.imshow(to_plot(image1))
plt.imshow(to_plot(image2))

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

mask_white = cv2.inRange(gray_image, 200, 255)
mask_white1 = cv2.inRange(gray_image1, 200, 255)
mask_white2 = cv2.inRange(gray_image2, 200, 255)

plt.imshow(mask_white)
plt.imshow(mask_white1[:113,:])
plt.imshow(mask_white1[113:,:])
plt.imshow(mask_white1)
plt.imshow(mask_white2)

title = "%d,%d"%(np.sum(mask_white1[:113,:]/255.0),np.sum(mask_white1[113:,:]/255.0))
title

# -------------------------------DQN AgentStart------------------------------------------------
def stock_action_log(action):
    data_index = cycle_of_episode%action_log_data_size
    action_log[data_index] = action
    print action_log

def get_initial_action():
    N0 =  len(np.where(action_log ==0)[0]) # 0の数を取得
    N1 =  len(np.where(action_log ==1)[0]) # 1の数を取得
    N2 =  len(np.where(action_log ==2)[0]) # 2の数を取得
    print u"直近の各アクションの数:",[N0,N1,N2]
    return np.argmax([N0,N1,N2])

action_log_data_size = 10
action_log = np.ones(action_log_data_size)
cycle_of_episode = 0
for i in range(30):
    action = np.random.randint(3)
    print action

    stock_action_log(action)

    action = get_initial_action()
    print action
    print "-----------------------------------------"
    cycle_of_episode += 1


print action_log
print (action_log ==2) # 2のindexがTrueになる
print np.where(action_log ==2 ) # 2のindexを取得(返り値Tupple)
print len(np.where(action_log ==2)[0]) # 2のindexを取得
actions = range(5)
np.zeros((len(actions)))
np.random.randint(3)

action_log_data_size = 15
action_log = np.empty(action_log_data_size, dtype=int)
action_log[:] = np.nan
