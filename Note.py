# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np

import matplotlib.pyplot as plt

from chainer import cuda, FunctionSet, Variable, optimizers, serializers
from chainer import cuda

# ----------QNetのinit----------
hist_size = 1 # いくつ前までの経験をStockするか

# cnn_dqn_agentでの変数
q_net_input_dim = 256 * 6 * 6 #=image_feature_dim (+ depth_image_dim)
# QNetでの変数
dim = q_net_input_dim

data_size = 10 #LIS -> 10**5
d = [np.zeros((data_size, hist_size, dim),
            dtype=np.uint8),
          np.zeros(data_size, dtype=np.uint8),
          np.zeros((data_size, 1), dtype=np.int8),
          np.zeros((data_size, hist_size, dim),
            dtype=np.uint8),
          np.zeros((data_size, 1), dtype=np.bool)]

len(d)
# state(data_size,hist_size,dim)
d[0].shape
# action
d[1].shape
# reward
d[2].shape
# state_dash(episode_end_flagがTrueの場合0が入る,data_size,hist_size,dim)
d[3].shape
# episode_end_flag(0 or 1)
d[4].shape


# ----------cnn_dqn_agentのagent_step(reward,observation)----------
#-----------rewardはメソッド内で使ってない->RoboCarServerでは消した-
obs_array = np.array(range(dim))
obs_array
obs_array.shape

state = np.asanyarray([obs_array], dtype=np.uint8)
#if hist_size==2 -> state=np.asanyarray([state[1],obs_array],dtype=np.uint8)
# ↑↑1つ前の経験も考慮
state_2hist = np.asanyarray([state[0],obs_array],dtype=np.uint8)
state_2hist.shape

# uint8
state.shape
# float32
state_ = np.asanyarray(state.reshape(1,hist_size,q_net_input_dim), dtype=np.float32)
state_.shape

#policy_frozenやinitial_explorationに合わせてepsを設定
#action, q_now = q_net.e_greedy(state_, eps)

action = 6
#--------------------agent_startメソッドの場合--------------------
last_action = copy.deepcopy(action)
last_state = state.copy()
last_hoge = copy.deepcopy(state)
last_state.shape
last_hoge.shape
#------------------------------------------------------------------

#return action, eps, q_now, obs_array
#serverがこのagent_stepのreturnにrewardを付け足して
#agent_step_updategent_step_update(reward,action,eps,q_now,obs_array)に渡す

reward = -1

# -----------------cnn_dqn_agentのagent_step_update-----------------
#q_netのstock_experience(time,last_state,last_action,reward,state,False)を呼ぶ
# Qnetのstock_experience(time,state,action,reward,state_dasg,episode_end_flag)
episode_end_flag = True
time = 16
data_index = time%data_size
data_index

d[0][data_index] = state
d[1][data_index] = action
d[2][data_index] = reward
#if(episode_end_flag == False)
d[4][data_index] = episode_end_flag
d[4]
#q_netのexperience_replay(time)を呼ぶ
replay_size = 5 # LIS -> 32
time
time2 = 7
time3 = 3
#if(time < data_size)
replay_index2 = np.random.randint(0,time2,(replay_size,1))
#else
replay_index = np.random.randint(0,data_size,(replay_size,1))

replay_index3 = np.random.randint(0,time3,(replay_size,1))

replay_index
replay_index2
replay_index3
s_replay = np.ndarray(shape=(replay_size, hist_size, dim), dtype=np.float32)
a_replay = np.ndarray(shape=(replay_size, 1), dtype=np.uint8)
r_replay = np.ndarray(shape=(replay_size, 1), dtype=np.float32)
s_dash_replay = np.ndarray(shape=(replay_size, hist_size, dim), dtype=np.float32)
episode_end_replay = np.ndarray(shape=(replay_size, 1), dtype=np.bool)

d[0].shape
np.array(d[0][replay_index[1]]).shape
s_replay.shape
s_replay[0].shape

d[1].shape
a_replay.shape
d[2].shape
r_replay.shape
d[3].shape
s_dash_replay.shape
d[4].shape
episode_end_replay.shape

for i in xrange(replay_size):
    s_replay[i] = np.asarray(d[0][replay_index[i]], dtype=np.float32)
    a_replay[i] = d[1][replay_index[i]]
    r_replay[i] = d[2][replay_index[i]]
    s_dash_replay[i] = np.array(d[3][replay_index[i]], dtype=np.float32)
    episode_end_replay[i] = d[4][replay_index[i]]

#-----------------------------実験-----------------------------
x_data = np.array(range(100,110)).reshape((1,1,10))+1
x_data2 = np.array(range(100,110)).reshape((1,10))+1
x_data.shape
x_data2.shape
x_replay = np.array(range(50)).reshape((5,1,10))+1
x_replay.shape
x_replay[0].shape
x_replay
x_replay[0] = x_data
x_replay[1] = x_data2
x_replay
# 揃ったreplay達をq_net.forwardへ送ってlossを得てoptimizer.update
# QNetのforward(s_replay,a_replay,r_replay,s_dash_replay,episode_end_replay))
num_of_batch = s_replay.shape[0]
num_of_batch
s = Variable(s_replay)
s_dash = Variable(s_dash_replay)
s.data.shape
# q->NNにsを入れた出力 tmp->NNにs_dashを入れた出力
q = np.random.rand(5,7)
tmp = np.random.rand(5,7)+1
q
tmp
map(np.max, tmp)
tmp = list(map(np.max, tmp))
tmp
max_q_dash = np.asanyarray(tmp, dtype=np.float32)
max_q_dash

target = np.array(q, dtype=np.float32)
target
#for i in range(num_of_batch):
#本来はfor文だがi=1の時だけやって省略
i = 1
tmp_ = r_replay[i] + 0.99*max_q_dash[i] # 0.99->割引率
tmp_
a_replay.shape
a_replay[i]
target[i,a_replay[i]]
target[i,a_replay[i]] = tmp_

target.shape
q.shape

td = target - q
abs(td)<=1
1000.0*(abs(td)<=1)


# -------------------------OpenCV-------------------------
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import numpy as np
import os
print 3
plt.gray()
def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#thick red lines

def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    global cache
    global first_frame
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    alpha =0.2
    #i got this alpha value off of the forums for the weighting between frames.
    #i understand what it does, but i dont understand where it comes from
    #much like some of the parameters in the hough function

    for line in lines:
        #1
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        #2
        y_global_min = min(y1,y2,y_global_min)

    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        print ('no lane detected')
        return 1

    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1



    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean)
    l_x2 = int((y_max - l_b)/l_slope_mean)
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_intercept)
        r_y1 = int((r_slope_mean * r_x1 ) + r_intercept)
        l_y2 = int((l_slope_mean * l_x2 ) + l_intercept)
        r_y2 = int((r_slope_mean * r_x2 ) + r_intercept)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype ="float32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else :
        prev_frame = cache
        next_frame = (1-alpha)*prev_frame+alpha*current_frame

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)

    cache = next_frame

#used below
def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

imgDir_path = '/Users/okuyamatakashi/RoboCar/RoboCarImage/'

file_list = os.listdir(imgDir_path)
for File in file_list:
    if '.jpg' not in File and '.jpeg' not in File and '.png' not in File and '.JPG' not in File:
        file_list.remove(File)

image = cv2.imread(imgDir_path+file_list[224])
image = cv2.imread('/Users/okuyamatakashi/desktop/sample.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image.shape
plt.gray()
plt.imshow(gray_image)
#src画像から指定した色の範囲にもとづいてマスク画像を生成する
mask_white = cv2.inRange(gray_image, 200, 255)
plt.imshow(mask_white)


#mask_image_not = cv2.bitwise_not(gray_image, mask_white)
#mask_image_or = cv2.bitwise_or(gray_image, mask_white)
mask_image = cv2.bitwise_and(gray_image, mask_white)
mask_image.shape
plt.imshow(mask_image)
# 平滑化
kernel_size = 5
gauss_gray = cv2.GaussianBlur(mask_image, (kernel_size, kernel_size), 0)
plt.imshow(gauss_gray)
# エッジ検出
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gauss_gray, low_threshold, high_threshold)
canny_edges.shape
plt.imshow(canny_edges)

# roi -> Region Of Interest
imgshape = image.shape
print imgshape
lower_left = [imgshape[1]/9,imgshape[0]]
lower_right = [imgshape[1]-imgshape[1]/9,imgshape[0]]
top_left = [imgshape[1]/2-imgshape[1]/8,imgshape[0]/2+imgshape[0]/10]
top_right = [imgshape[1]/2+imgshape[1]/8,imgshape[0]/2+imgshape[0]/10]

lower_left = [0,100]
lower_right = [227,100]
top_left = [75,0]
top_right = [227-75,0]
vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

print lower_left
print lower_right
print top_left
print top_right
print vertices[0].shape
#defining a blank mask to start with
roi_mask = np.zeros_like(mask_white)

#filling pixels inside the polygon defined by "vertices" with the fill color
cv2.fillPoly(roi_mask, vertices, 255)
plt.imshow(roi_mask)
roi_image = mask_white.copy()
roi_image = cv2.bitwise_and(mask_white, roi_mask)
plt.imshow(roi_image)

#rho and theta are the distance and angular resolution of the grid in Hough space
#same values as quiz
rho = 4
theta = np.pi/180
#threshold is minimum number of intersections in a grid for candidate line to go to output
threshold = 30
min_line_len = 100
max_line_gap = 180
#my hough values started closer to the values in the quiz, but got bumped up considerably for the challenge video
lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
lines

plt.imshow(to_plot(image))
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)

plt.imshow(to_plot(image))

line_img = np.zeros((canny_edges.shape[0], canny_edges.shape[1], 3), dtype=np.uint8)


line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)

draw_lines(line_img,lines)

def make_detection_image(img,mask):
    flip_mask = 255-mask
    img[:,:,0] = cv2.bitwise_and(img[:,:,0],img[:,:,0], mask=flip_mask)
    img[:,:,1] = cv2.bitwise_and(img[:,:,1],img[:,:,1], mask=flip_mask)
    return img


imgDir_path = '/Users/okuyamatakashi/RoboCar/RoboCarImage/'
file_list = os.listdir(imgDir_path)
len(file_list)
for File in file_list:
    if '.jpg' not in File and '.jpeg' not in File and '.png' not in File and '.JPG' not in File:
        file_list.remove(File)
i = 0
image1 = cv2.imread(imgDir_path+file_list[90])
image2 = cv2.imread(imgDir_path+file_list[20])
image1 = cv2.resize(image1,(227,227))
image2 = cv2.resize(image2,(227,227))
plt.imshow(to_plot(image2))
plt.imshow(to_plot(image1))

image1 = cv2.resize(image1,(227,227))
image2 = cv2.resize(image2,(227,227))
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

mask_white1 = cv2.inRange(gray_image1, 200, 255)
mask_white2 = cv2.inRange(gray_image2, 200, 255)

plt.imshow((mask_white2*mask)/255.0)

img = cv2.imread('/Users/okuyamatakashi/desktop/00000.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_flip = cv2.flip(img,1)
plt.imshow(img)
plt.imshow(img_flip)
mask = (img+img_flip)/2
plt.imshow(mask)
hl = 0
hr = 0
mask[i,226]
while(mask[hl,0]==0):
    hl+=1
while(mask[hr,226]==0):
    hr+=1
hl
hr
cv2.imwrite("/Users/okuyamatakashi/desktop/hoge.png",mask)
