# -*- coding: utf-8 -*-
import six.moves.cPickle as pickle
import copy
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import zmq
#%matplotlib inline
import argparse

from DQN import CnnDqnAgent
#from Chen import
#from Adati import

parser = argparse.ArgumentParser(description='ml-agent-for-unity')

parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help=u'GPU ID (negative value indicates CPU)')

parser.add_argument('--folder', '-f', default='ModelGod', type=str,
                    help=u'モデルの存在するフォルダ名')
parser.add_argument('--model_num', '-m', default=600000,type=int,
                    help=u'最初にロードするモデルの番号')

parser.add_argument('--NN', '-n', action = "store_true",
                    help=u'ニューラルネットワークを使うか')
parser.add_argument('--adati', '-a', action = "store_true",
                    help=u'足立くんのNNを使ってLaneDetectionするか')
parser.add_argument('--chen', '-c', action = "store_true",
                    help=u'ChenくんのNNを使って信号認識するか')

parser.add_argument('--image', '-i', action = "store_true",
                    help=u'imageをPlotするか')
parser.add_argument('--q_value', '-q', action = "store_true",
                    help=u'Q_ValueをPlotするか')

parser.add_argument('--test', '-t', action = "store_true",
                    help=u'TEST frags, False => Train')
args = parser.parse_args()

if(args.image and args.q_value):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.gray()
elif(args.image):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()
elif(args.q_value):
    fig, ax3 = plt.subplots(1, 1)


def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed

def lane_detection(img):
    img = img.reshape((600,800,3))
    img_gray = to_grayscale(img)

    img_resize = cv2.resize(img_gray,(227,227))

    kernel = np.ones((10,10),np.float32)/100
    img_resize = cv2.filter2D(img_resize,-1,kernel)
    img_threshold = cv2.threshold(img_resize,180,255,cv2.THRESH_BINARY)[1]
    #img_cutblack = cv2.rectangle(img_threshold,(0,0),(227,10),(0,0,0),-1)

    return img_threshold

def pause_Image_plot(img1,img2):
    ax1.cla()
    ax1.imshow(img1)
    ax2.cla()
    ax2.imshow(img2)
#Q関数のplot
def pause_Q_plot(q):
    ax2.cla()
    actions = range(7)

    max_q_abs = max(abs(q))
    if max_q_abs != 0:
        q = q / float(max_q_abs)

    ax2.set_xticks(actions)
    ax2.set_xticklabels(['-30','-20','-10','0','10','20','30'], rotation=0, fontsize='small')
    ax2.set_xlabel("Action") # x軸のラベル
    ax2.set_ylabel("Q_Value") # y軸のラベル
    ax2.set_ylim(-1.1, 1.1)  # yを-1.1-1.1の範囲に限定
    ax2.set_xlim(-1, 7) # xを-0.5-7.5の範囲に限定
    ax2.hlines(y=0, xmin=-37, xmax=37, colors='r', linewidths=2) #y=0の直線

    ax2.bar(actions,q,align="center")

def decide_test_action(action,q):
    q_max = q.ravel()[action]
    q_3 = q.ravel()[3]
    #print "q_max : %f"%(q_max)
    #print "q_3 : %f"%(q_3)
    #print "q_3 / q_max : %f"%(q_3 / q_max)
    #print "q_max - q_3 : %f"%(q_max - q_3)
    if(q_max != 0):
        divide = q_3 / q_max
    else:
        divide = 1
    if(divide > 0.97):
        return 3
    else:
        return action
    return action
def send_action(action):
    #  Send reply back to client
    socket_local.send(np.array([action]))
    print "Send Action : %d"%(action)



if __name__ == '__main__':
    # model_simのQ_MAXがこの値より低かったらEpisode終了
    death_value = 10

    gpu = args.gpu

    folder = args.folder
    model_num = args.model_num

    NN = args.NN
    use_adati = args.adati
    use_chen = args.chen

    test = args.test

    death = True
    time = 0

    if(NN):
        agent = CnnDqnAgent();
        agent.agent_init(
            use_gpu = gpu,
            folder = folder,
            model_num = model_num
        )
    if(use_adati):
        adati = adati君のクラス
    if(use_chen):
        chen = chen君のクラス

    #-----------------logファイル作成----------------------------

    context1 = zmq.Context()
    socket_local = context1.socket(zmq.REP)
    socket_local.bind("tcp://*:5555")

    print "Waiting Request..."

    while True:
        # Receve Data from C++ Program
        data =  socket_local.recv()
        print "Received"
        image = np.frombuffer(data, dtype=np.uint8);

        # -------------------ここより上は月曜日修正-----------------------

        if(use_chen and chen.hogehoge(image1)):
            send_action(100)

        else:
            if(use_adati)
                detection_image = adati.hogehoge(image2)
                detection_image = detection_image*255
                image2_g = cv2.resize(detection_image,(227,227))
            elif:
                image2_g = lane_detection(image)
            new_image2 = cv2.merge((image_g,image_g,image_g))

            if(NN):
                if(death):
                    death,test_action,test_q = agent.check_death(new_image2,death_value,test)
                    if(death):
                        print "Agent is Death"
                        action = 7
                    else:
                        print "Episode START"
                        if(test):
                            action = decide_test_action(test_action,test_q)
                        else:
                            action = agent.agent_start(new_image2)
                        episode_start_time = time
                        time += 1
                    send_action(action)

                else:
                    death,test_action,test_q = agent.check_death(new_image2,death_value,test)
                    if(death):
                        send_action(7)
                        score = time - episode_start_time
                        if(not test):
                            # --------------logファイルへの書き込み--------------
                            reward = -1
                            agent.agent_end(reward,time)
                        print "Score is %d"%(score)
                    else:
                        if(test):
                            action = decide_test_action(test_action,test_q)
                            send_action(action)
                        else:
                            action, q_now, obs_array = agent.agent_step(new_image2)
                            send_action(action)
                            reward = 0
                            agent.agent_step_update(reward,time,action,q_now,obs_array)
                        time += 1
            else:
                q = np.random.rand(13)
                send_action(6)

        if(args.image or args.q_value):
            if(args.image):
                pause_Image_plot(image1,new_image2)
            elif(args.q_value):
                print q.ravel()
                pause_Q_plot(q.ravel())
            plt.pause(1.0 / 10**10) #引数はsleep時間

        print "------------------------------"
