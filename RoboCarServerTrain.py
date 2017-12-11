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

parser.add_argument('--folder', '-f', default='ModelRobo', type=str,
                    help=u'モデルの存在するフォルダ名')
parser.add_argument('--model_num', '-m', default=520000,type=int,
                    help=u'最初にロードするモデルの番号')

parser.add_argument('--NN', '-n', action = "store_true",
                    help=u'ニューラルネットワークを使うか')
parser.add_argument('--edit', '-e', action = "store_true",
                    help=u'エディットモードにするかZeroMQ通信にするか')


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

        model_num = options['model_num']
        #save_modelでもしようするため,selfをつけた
        self.folder = options["folder"]
        self.time = 0
        self.epsilon = 0

        self.q_net.load_model(folder,model_num)

    # 行動取得系,state更新系メソッド
    def agent_step(self, image):
        obs_array = self.feature_extractor.feature(image)

        self.state = np.asanyarray([obs_array], dtype=np.uint8)

        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Generate an Action by e-greedy action selection
        action, q_now = self.q_net.e_greedy(state_, 0)
#--------------------------------------------------------------------------
        # Simple text based visualization
        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)
        print('Step:%d  Action:%d  Q_max:%3f' % (self.time, self.q_net.action_to_index(action), q_max))
        # Time count
        self.time += 1
#--------------------------------------------------------------------------
        #return action, eps, q_now, obs_array
        return action, q_now

   # 学習系メソッド
    def agent_step_update(self, reward, action, eps, q_now, obs_array):
        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.last_action, reward, self.state, False)
            self.q_net.experience_replay(self.time)

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Simple text based visualization
        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)

        print('Step:%d  Action:%d  Reward:%.1f  Epsilon:%.6f  Q_max:%3f' % (
            self.time, self.q_net.action_to_index(action), reward, eps, q_max))

        # Updates for next step , 更新するだけで使ってない
        self.last_observation = obs_array

        if self.policy_frozen is False:
            self.last_action = copy.deepcopy(action)
            self.last_state = self.state.copy()

            # save model
            if self.q_net.initial_exploration < self.time and np.mod(self.time,self.q_net.save_model_freq) == 0:
                print "------------------Save Model------------------"
                self.q_net.save_model(self.folder,self.time)

        # Time count
        self.time += 1
args = parser.parse_args()

if __name__ == '__main__':
    gpu = args.gpu
    folder = args.folder
    model_num = args.model_num
    NN = args.NN
    edit = args.edit

    if(NN):
        agent = CnnDqnAgent();
        agent.agent_init(
            use_gpu = gpu,
            folder = folder,
            model_num = model_num
        )

    if(not edit):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        print "Waiting Request..."

    i = 0
    while True:
        if(edit):
            if(i == 0):
                image_list = make_image_list()
            elif(i>150):
                i = 0
            image = image_list[i]

        else:
            # Receve Data from C++ Program
            data =  socket.recv()
            print "Received Request"
            BGR_image = np.frombuffer(data, dtype=np.uint8).reshape((227,227,3));
            #image = to_plot(BGR_image)
            image = BGR_image


        new_image = BackGround2Black(image)

        if(NN):
            action, q = agent.agent_step(image)
        else:
            #action = np.random.randint(7)
            action = 3
            q = np.random.rand(7)

        pause_Image_plot(image)
        #pause_Q_plot(q.ravel())

        if(not edit):
            #  Do some 'work'
            #time.sleep(0.05)
            #action = np.array( [action] )
            #  Send reply back to client
            socket.send(np.array([action]))
            print "Send Action : %d"%(action)

        i+=1
