# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np

import matplotlib.pyplot as plt

from chainer import cuda

from cnn_feature_extractorRobo import CnnFeatureExtractor
from q_netRobo import QNet

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
    def agent_start(self, image):
        print "----------------------------------"
        print "RoboCar Start"
        print "----------------------------------"
        obs_array = self.feature_extractor.feature(image)
        # Initialize State
        self.state = np.zeros((self.q_net.hist_size, self.q_net_input_dim), dtype=np.uint8)
        self.state[0] = obs_array
        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Generate an Action e-greedy
        action, q_now = self.q_net.e_greedy(state_, 0)

        return action

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
        print('Step:%d  Action:%d  Q_max:%3f' % (self.time, self.q_net.action_to_index(action), q_max))

        # Time count
        self.time += 1

        return action, q_now

    def received_image(self, image):
        if not self.agent_initialized:
            self.agent_initialized = True
            print ("initializing agent...")
            #depth_image_dimが引数で使われるのはここだけ
            self.agent_init(
                use_gpu=self.gpu,
                folder = self.folder,
                model_num = self.model_num)

            action = self.agent_start(image)
            #self.send_action(action)

        else:
            action, q_now = self.agent_step(image)

        return action

print 1

range(9)+1
a = (np.array(range(9),dtype = np.uint8)+1).reshape((3,3))

a
np.max(a)

c = a.astype(np.float32)
c
b = np.frombuffer(a.data, dtype=np.uint);
b
