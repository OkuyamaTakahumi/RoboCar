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
#from AdatiDir.Adati import LaneDetection

parser = argparse.ArgumentParser(description='ml-agent-for-unity')

parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help=u'GPU ID (negative value indicates CPU)')

parser.add_argument('--folder', '-f', default='ModelRobo3Real', type=str,
                    help=u'モデルの存在するフォルダ名')
parser.add_argument('--model_num', '-m', default=600000,type=int,
                    help=u'最初にロードするモデルの番号')

parser.add_argument('--NN', '-n', action = "store_true",
                    help=u'ニューラルネットワークを使うか')
parser.add_argument('--adati', '-a', action = "store_true",
                    help=u'足立くんのNNを使ってLaneDetectionするか')

parser.add_argument('--image', '-i', default=0,type=int,
                    help=u'plotするImageの枚数')
parser.add_argument('--q_value', '-q', action = "store_true",
                    help=u'Q_ValueをPlotするか')

parser.add_argument('--test', '-t', action = "store_true",
                    help=u'TEST frags, False => Train')
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help=u'reward log file name')
args = parser.parse_args()

if(args.image==2 and args.q_value):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.gray()
elif(args.image==1 and args.q_value):
    fig, (ax1, ax3) = plt.subplots(1, 2)
    plt.gray()
elif(args.image==2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()
elif(args.image==1):
    fig, ax1 = plt.subplots(1, 1)
    plt.gray()
elif(args.q_value):
    fig, ax3 = plt.subplots(1, 1)


def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed

# reshape済みの配列が引数,3chanelのimgをreturn
def lane_detection(img):
    img_gray = to_grayscale(img)

    img_resize = cv2.resize(img_gray,(227,227))

    kernel = np.ones((10,10),np.float32)/100
    img_resize = cv2.filter2D(img_resize,-1,kernel)
    img_threshold = cv2.threshold(img_resize,180,255,cv2.THRESH_BINARY)[1]
    #img_threshold = cv2.rectangle(img_threshold,(0,0),(227,10),(0,0,0),-1)
    return cv2.merge((img_threshold,img_threshold,img_threshold))

# def pause_Image_plot(img1,img2):
def pause_Image_plot1(img1):
    ax1.cla()
    #ax1.tick_params(labelleft="off",labelbottom='off')
    #ax1.title.set_text('Lane Detection Image')
    ax1.imshow(img1)

def pause_Image_plot2(img1,img2):
    ax1.cla()
    ax2.cla()
    #ax1.tick_params(labelleft="off",labelbottom='off')
    #ax2.tick_params(labelleft="off",labelbottom='off')
    #ax1.title.set_text('Original Image')
    #ax2.title.set_text('Lane Detection Image')
    ax1.imshow(img1)
    ax2.imshow(img2)

#Q関数のplot
def pause_Q_plot(q, a_num):
    ax3.cla()
    #ax3.title.set_text('Q_values of each action')
    actions = range(a_num)

    max_q_abs = max(abs(q))
    if max_q_abs != 0:
        q = q / float(max_q_abs)

    ax3.set_xticks(actions)
    if(a_num==7):
        ax3.set_xticklabels(['-30','-20','-10','0','10','20','30'], rotation=0, fontsize='small')
    elif(a_num==13):
        ax3.set_xticklabels(['-30','-25','-20','-15','-10','-5','0','5','10','15','20','25','30'], rotation=0, fontsize='small')
    ax3.set_xlabel("Action") # x軸のラベル
    ax3.set_ylabel("Q_Value") # y軸のラベル
    ax3.set_ylim(-1.1, 1.1)  # yを-1.1-1.1の範囲に限定
    ax3.set_xlim(-1, a_num) # xを-0.5-7.5の範囲に限定
    ax3.hlines(y=0, xmin=-1, xmax=a_num, colors='r', linewidths=2) #y=0の直線

    ax3.bar(actions,q,align="center")

def decide_test_action(action,q,a_num):
    q_max = q.ravel()[action]
    forward_index = (a_num-1)/2
    q_forward = q.ravel()[forward_index]
    print "q : "
    print q.ravel()
    print "q_max : %f"%(q_max)
    print "action : %d"%(action)
    #print "q_forward : %f"%(q_forward)
    #print "q_forward / q_max : %f"%(q_forward / q_max)
    #print "q_max - q_forward : %f"%(q_max - q_forward)
    if(q_max != 0):
        divide = q_forward / q_max
    else:
        divide = 1
    if(divide > 0.97):
        print "action : %d"%(forward_index)
        return forward_index,q_max
    else:
        return action,q_max

def send_action(action):
    #  Send reply back to client
    socket.send(np.array([action]))
    #print "Send Action : %d"%(action)



if __name__ == '__main__':
    # model_simのQ_MAXがこの値より低かったらEpisode終了
    death_value = 10

    gpu = args.gpu
    use_adati = args.adati
    folder = args.folder
    model_num = args.model_num
    NN = args.NN
    test = args.test
    log_file = args.log_file

    death = True
    time = 0
    episode_num = 1
    score = 0

    if(use_adati):
        print "Use Adati's NN"
        adati = LaneDetection()

    a_num = 13
    hidden_dim = 256
    if(folder=="ModelRobo3Real" or folder=="ModelRobo4Real"):
        a_num = 7
    elif(folder=="ModelRobo5_2Real" or folder=="ModelRobo6_2Real"):
        hidden_dim = 512
    else:
        print u"There is not \"%s\" folder"%(folder)
        import sys
        sys.exit()

    if(NN):
        agent = CnnDqnAgent();
        agent.agent_init(
            use_gpu = gpu,
            folder = folder,
            model_num = model_num,
            a_num = a_num,
            hidden_dim = hidden_dim,
            test = test
        )

    # logファイル作成
    if(not test):
        with open(log_file, 'w') as the_file:
            the_file.write('Cycle,Score,Episode \n')

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    print "Waiting Request..."

    while True:
        # Receve Data from C++ Program
        data = socket.recv()
        #data1, data2=  socket.recv_multipart()
        #print "Received"

        image = np.frombuffer(data, dtype=np.uint8);
        image = image.reshape((600,800,3))

        #if(use_adati):
            #detection_image = adati.mainfunction(image)
            #new_image = cv2.resize(detection_image,(227,227))
        #else:
        new_image = lane_detection(image)

        if(NN):
            if(death):
                death,test_action,test_q = agent.check_death(new_image,death_value)

                if(death):
                    print "Agent is Death"
                    action = 100
                else:
                    print "Episode %d START"%(episode_num)
                    if(test):
                        action,q_max = decide_test_action(test_action,test_q,a_num)
                    else:
                        action = agent.agent_start(new_image)
                    episode_start_time = time
                    time += 1
                send_action(action)

            else:
                death,test_action,test_q = agent.check_death(new_image,death_value)

                if(death):
                    ep_score = time - episode_start_time
                    print "Episode Score is %d"%(ep_score)
                    action = 100 # back
                    if(not test):
                        score += ep_score
                        if(episode_num%10 == 0):
                            print "Finish %d Episode"%(episode_num)
                            action = 101 #stop
                            # logファイルへの書き込み
                            with open(log_file, 'a') as the_file:
                                the_file.write(str(time)+','+str(score) +','+str(episode_num) + '\n')
                            print "Score is %d"%(score)
                            score = 0
                        send_action(action)
                        reward = -1
                        agent.agent_end(reward,time)
                    else:
                        send_action(action)
                    episode_num += 1

                else:
                    if(test):
                        action,q_max = decide_test_action(test_action,test_q,a_num)
                        send_action(action)
                    else:
                        action, q_now = agent.agent_step(new_image)

                        send_action(action)
                        reward = 0
                        agent.agent_step_update(reward,time,action,q_now)
                    time += 1
        else:
            q = np.random.rand(a_num)
            action = 3
            action,q_max = decide_test_action(action,q,a_num)
            send_action(action)

        if(args.image>0 or args.q_value):
            #if(test):
                #fig.suptitle("Q_Max : %f"%(q_max),fontsize=24)
            if(args.image==1):
                pause_Image_plot1(new_image)
            elif(args.image==2):
                pause_Image_plot2(to_plot(image),new_image)
            if(args.q_value):
                pause_Q_plot(q.ravel(),a_num)
            plt.pause(1.0 / 10**10) #引数はsleep時間
            #plt.pause(1.0 / 10**10) #引数はsleep時間

        print "---------------------------------------------------------"
