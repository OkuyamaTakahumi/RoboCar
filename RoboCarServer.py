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
from q_netRobo import QNet
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


class CnnDqnAgent(object):
    #アクションリスト(数字じゃなくても大丈夫)
    actions = [0, 1, 2, 3, 4, 5, 6]

    cnn_feature_extractor = 'alexnet_feature_extractor.pickle' #1
    model = 'bvlc_alexnet.caffemodel' #2
    model_type = 'alexnet' #3

    # AlexNetの出力
    image_feature_dim = 256 * 6 * 6

    time = 0

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

        #save_modelでもしようするため,selfをつけた
        folder = options["folder"]

        model_num = options['model_num']

        self.q_net.load_model(folder,model_num)

    # 行動取得系,state更新系メソッド
    def agent_step(self, image):
        obs_array = self.feature_extractor.feature(image)

        # Compose State : 4-step sequential image
        if self.q_net.hist_size == 4:
            self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_array], dtype=np.uint8)
        elif self.q_net.hist_size == 2:
            self.state = np.asanyarray([self.state[1], obs_array], dtype=np.uint8)
        elif self.q_net.hist_size == 1:
            self.state = np.asanyarray([obs_array], dtype=np.uint8)
        else:
            print("self.DQN.hist_size err")

        # q_funcに入れる際は(サンプル数,hist_size,q_net_input_dim)
        # 例:np.ndarray(shape=(self.replay_size, self.hist_size, self.dim))
        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Generate an Action by e-greedy action selection
        action, q_now = self.q_net.e_greedy(state_, 0)

        # Simple text based visualization
        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)


        if(q_max < 5):
            action = 7




        print('Step:%d  Action:%d  Q_max:%3f' % (self.time, action, q_max))

        q_diff = q_max - q_now.ravel()[3]
        print(q_now.ravel())
        print("Q_MAX - Q of action4 = %f" %(q_diff))


        # Time count
        self.time += 1

        return action, q_now





if __name__ == '__main__':
    gpu = args.gpu
    adati = args.adati
    folder = args.folder
    model_num = args.model_num
    NN = args.NN

    if(NN):
        agent = CnnDqnAgent();
        agent.agent_init(
            use_gpu = gpu,
            folder = folder,
            model_num = model_num
        )

    context1 = zmq.Context()
    socket_local = context1.socket(zmq.REP)
    socket_local.bind("tcp://*:5555")

    if(adati):
        context2 = zmq.Context()
        socket_to_adati = context2.socket(zmq.REQ)
        socket_to_adati.connect("tcp://192.168.11.16:5556")
    print "Waiting Request..."

    i = 0
    while True:
        # Receve Data from C++ Program
        data =  socket_local.recv()
        print "Received RoboCar's original Image"
        BGR_image = np.frombuffer(data, dtype=np.uint8);
        RGB_image = to_plot(BGR_image)

        if(adati):
            socket_to_adati.send(RGB_image.data)
            data =  socket_to_adati.recv()
            print "Received Lane Detection Image"
            detection_image = np.frombuffer(data, dtype=np.uint8).reshape((224,224));
            detection_image = detection_image*255
            new_image_g = cv2.resize(detection_image,(227,227))

        else:
            new_image_g = lane_detection(RGB_image)

        new_image = cv2.merge((new_image_g,new_image_g,new_image_g))

        if(NN):
            action, q = agent.agent_step(new_image)
            #print q.ravel()
        else:
            action = 1
            q = np.zeros((7))
            q[action] = 1

        if(args.image or args.q_value):
            if(args.image):
                pause_Image_plot(new_image)
            elif(args.q_value):
                print q.ravel()
                pause_Q_plot(q.ravel())
            plt.pause(1.0 / 10**10) #引数はsleep時間


        #  Do some 'work'
        #time.sleep(0.05)
        #  Send reply back to client
        socket_local.send(np.array([action]))
        print "Send Action : %d"%(action)
        print "---------------------------------------------------------------"

        i+=1
