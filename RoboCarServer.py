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

from chainer import cuda

from cnn_feature_extractorRobo import CnnFeatureExtractor
from q_netRoboCar import QNet
import argparse

parser = argparse.ArgumentParser(description='ml-agent-for-unity')

parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help=u'GPU ID (negative value indicates CPU)')

parser.add_argument('--adati', '-a', action = "store_true",
                    help=u'足立くんのNNを使ってLaneDetectionするか')

parser.add_argument('--folder', '-f', default='ModelGod', type=str,
                    help=u'モデルの存在するフォルダ名')
parser.add_argument('--model_num', '-m', default=600000,type=int,
                    help=u'最初にロードするモデルの番号')

parser.add_argument('--NN', '-n', action = "store_true",
                    help=u'ニューラルネットワークを使うか')

parser.add_argument('--image', '-i', action = "store_true",
                    help=u'imageをPlotするか')
parser.add_argument('--q_value', '-q', action = "store_true",
                    help=u'Q_ValueをPlotするか')

parser.add_argument('--test', '-t', action = "store_true",
                    help=u'TEST frags, False => Train')
args = parser.parse_args()

if(args.image and args.q_value):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()
elif(args.image):
    fig, ax1 = plt.subplots(1, 1)
    plt.gray()
elif(args.q_value):
    fig, ax2 = plt.subplots(1, 1)


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

def pause_Image_plot(img):
    ax1.cla()
    ax1.imshow(img)
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

class CnnDqnAgent(object):
    #アクションリスト(数字じゃなくても大丈夫)
    actions = [0, 1, 2, 3, 4, 5, 6]

    cnn_feature_extractor = 'alexnet_feature_extractor.pickle' #1
    model = 'bvlc_alexnet.caffemodel' #2
    model_type = 'alexnet' #3

    # AlexNetの出力
    image_feature_dim = 256 * 6 * 6

    agent_initialized = False

    def agent_init(self, **options):
        print ("initializing agent...")
        self.use_gpu = options['use_gpu']
        self.q_net_input_dim = self.image_feature_dim

        if os.path.exists(self.cnn_feature_extractor):
            print("loading... " + self.cnn_feature_extractor),
            self.feature_extractor = pickle.load(open(self.cnn_feature_extractor))
            print("done")

        # 初めてLISを起動する時
        else:
            self.feature_extractor = CnnFeatureExtractor(self.use_gpu, self.model, self.model_type, self.image_feature_dim)
            pickle.dump(self.feature_extractor, open(self.cnn_feature_extractor, 'w'))
            print("pickle.dump finished")

        self.q_net = QNet(self.use_gpu, self.actions, self.q_net_input_dim)
        self.q_net_sim = QNet(self.use_gpu, self.actions, self.q_net_input_dim)

        model_num = options['model_num']
        #save_modelでもしようするため,selfをつけた
        self.folder = options["folder"]

        self.q_net.load_model(folder,model_num)
        self.q_net_sim.load_model(folder,0)

    # 行動取得系,state更新系メソッド
    def agent_start(self, image):
        #obs_array = self.feature_extractor.feature(image)
        # Initialize State
        #self.state = np.asanyarray([obs_array], dtype=np.uint8)
        #self.state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)

        #if self.use_gpu >= 0:
            #self.state_ = cuda.to_gpu(self.state_)

        # Generate an Action e-greedy
        action, q_now = self.q_net.e_greedy(self.state_, 0)
        return_action = action

        # Update for next step
        self.last_action = copy.deepcopy(return_action)
        self.last_state = self.state.copy()
        print "Record last State and Action"

        return return_action

    # 学習系メソッド
    def agent_end(self, reward, time):  # Episode Terminated
        # Learning Phase
        self.q_net.stock_experience(time, self.last_state, self.last_action, reward, self.last_state,True)
        self.q_net.experience_replay(time)

        # Target model update
        if np.mod(time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        print('episode finished. Reward:%.1f' % (reward))

        # Model Save
        if np.mod(time,self.q_net.save_model_freq) == 0:
            print "------------------Save Model------------------"
            self.q_net.save_model(self.folder,time)

    # 行動取得系,state更新系メソッド
    def agent_step(self, image):
        #obs_array = self.feature_extractor.feature(image)

        #self.state = np.asanyarray([obs_array], dtype=np.uint8)

        #self.state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        #if self.use_gpu >= 0:
            #self.state_ = cuda.to_gpu(self.state_)

        # Generate an Action by e-greedy action selection
        action, q_now = self.q_net.e_greedy(self.state_, 0)

        #return action, eps, q_now, obs_array
        return action, q_now, obs_array

    # 学習系メソッド
    def agent_step_update(self, reward, time, action, q_now, obs_array):
        # Learning Phase
        self.q_net.stock_experience(time, self.last_state, self.last_action, reward, self.state, False)
        self.q_net.experience_replay(time)

        # Target model update
        if np.mod(time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Simple text based visualization
        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)

        print('Step:%d  Action:%d  Reward:%.1f Q_max:%3f' % (
            time, self.q_net.action_to_index(action), reward, q_max))

        self.last_action = copy.deepcopy(action)
        self.last_state = self.state.copy()
        print "Record last State and Action"

        # save model
        if np.mod(time,self.q_net.save_model_freq) == 0:
            print "------------------Save Model------------------"
            self.q_net.save_model(self.folder,time)

    def check_death(self,image,death_value,test):
        # image -> obs_array
        obs_array = self.feature_extractor.feature(image)
        self.state = np.asanyarray([obs_array], dtype=np.uint8)
        # obs_array -> NN -> get Q_MAX
        self.state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            self.state_ = cuda.to_gpu(self.state_)

        # Generate an Action by e-greedy action selection
        #q_max = self.q_net.sim_q_func(self.state_)
        if(test):
            action,q = self.q_net.e_greedy(self.state_, 0)
        else:
            action,q = self.q_net_sim.e_greedy(self.state_, 0)

        q_max = q.ravel()[action]
        print "Model_Sim Q_MAX:%3f"%(q_max)
        death = q_max < death_value

        print "Chack Death : %r"%(death)
        return death, action, q


if __name__ == '__main__':
    # model_simのQ_MAXがこの値より低かったらEpisode終了
    death_value = 10

    gpu = args.gpu
    adati = args.adati
    folder = args.folder
    model_num = args.model_num
    NN = args.NN
    test = args.test

    death = True
    time = 0

    agent = CnnDqnAgent();
    agent.agent_init(
        use_gpu = gpu,
        folder = folder,
        model_num = model_num
        )
    #-----------------logファイル作成----------------------------

    context1 = zmq.Context()
    socket_local = context1.socket(zmq.REP)
    socket_local.bind("tcp://*:5555")

    if(adati):
        context2 = zmq.Context()
        socket_to_adati = context2.socket(zmq.REQ)
        socket_to_adati.connect("tcp://192.168.11.16:5556")
    print "Waiting Request..."

    while True:
        # Receve Data from C++ Program
        data =  socket_local.recv()
        print "Received"
        image = np.frombuffer(data, dtype=np.uint8);

        if(adati):
            socket_to_adati.send(image.data)
            data =  socket_to_adati.recv()
            print "Received Lane Detection Image"
            detection_image = np.frombuffer(data, dtype=np.uint8).reshape((224,224));
            detection_image = detection_image*255
            new_image_g = cv2.resize(detection_image,(227,227))

        else:
            new_image_g = lane_detection(image)

        new_image = cv2.merge((new_image_g,new_image_g,new_image_g))

        if(death):
            death,test_action,test_q = agent.check_death(new_image,death_value,test)
            if(death):
                print "Agent is Death"
                action = 7
            else:
                print "Episode START"
                if(test):
                    action = decide_test_action(test_action,test_q)
                else:
                    action = agent.agent_start(new_image)
                episode_start_time = time
                time += 1
            send_action(action)

        else:
            death,test_action,test_q = agent.check_death(new_image,death_value,test)
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
                    action, q_now, obs_array = agent.agent_step(new_image)
                    send_action(action)
                    reward = 0
                    agent.agent_step_update(reward,time,action,q_now,obs_array)
                time += 1

        if(args.image or args.q_value):
            if(args.image):
                pause_Image_plot(new_image)
            elif(args.q_value):
                print q.ravel()
                pause_Q_plot(q.ravel())
            plt.pause(1.0 / 10**10) #引数はsleep時間

        print "------------------------------"
