# -*- coding: utf-8 -*-
import six.moves.cPickle as pickle
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda

from cnn_feature_extractorRoboCar import CnnFeatureExtractor
from q_netRoboCar_new import QNet

class CnnDqnAgent(object):
    cnn_feature_extractor = 'alexnet_feature_extractor.pickle' #1
    model = 'bvlc_alexnet.caffemodel' #2
    model_type = 'alexnet' #3

    # AlexNetの出力
    q_net_input_dim = 256 * 6 * 6
    agent_initialized = False
    q_max_max = 0.0

    def agent_init(self, **options):
        self.use_gpu = options['use_gpu']
        test = options['test']
        self.folder = options["folder"] #save_modelで使う->self.
        model_num = options['model_num']
        self.actions = range(options['a_num'])#数字じゃなくても大丈夫

        print "folder = %s"%(self.folder)
        print "actions = ",
        print self.actions


        if os.path.exists(self.cnn_feature_extractor):
            print("loading... " + self.cnn_feature_extractor),
            self.feature_extractor = pickle.load(open(self.cnn_feature_extractor))
            print("done")

        # 初めてLISを起動する時
        else:
            self.feature_extractor = CnnFeatureExtractor(self.use_gpu, self.model, self.model_type, self.q_net_input_dim)
            pickle.dump(self.feature_extractor, open(self.cnn_feature_extractor, 'w'))
            print("pickle.dump finished")

        self.q_net = QNet(self.use_gpu,self.actions,self.q_net_input_dim)
        self.q_net_sim = QNet(self.use_gpu,self.actions,self.q_net_input_dim)


        self.q_net.load_model(self.folder,model_num)
        if(test):
            self.q_net_sim.load_model(self.folder,model_num)
        else:
            self.q_net_sim.load_model(self.folder,0)

    # 行動取得系,state更新系メソッド
    def agent_start(self, image):
        obs_array = self.feature_extractor.feature(image)
        # Initialize State
        self.state = np.zeros((self.q_net.hist_size, self.q_net_input_dim), dtype=np.uint8)
        self.state[0] = obs_array
        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Generate an Action e-greedy
        action, q_now = self.q_net.e_greedy(state_, 0)
        return_action = action

        # Update for next step
        self.last_action = copy.deepcopy(return_action)
        self.last_state = self.state.copy()
        print "Record last State and Action"

        return return_action, q_now

    # 学習系メソッド
    def agent_end(self, reward, time):  # Episode Terminated
        # Learning Phase
        self.q_net.stock_experience(time, self.last_state, self.last_action, reward, self.last_state,True)
        self.q_net.experience_replay(time)

        # Target model update
        if np.mod(time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        print('episode finished Reward:%.1f' % (reward))

        # Model Save
        if np.mod(time,self.q_net.save_model_freq) == 0:
            print "------------------Save Model------------------"
            self.q_net.save_model(self.folder,time)

    # 行動取得系,state更新系メソッド
    def agent_step(self, image):
        obs_array = self.feature_extractor.feature(image)

        # Compose State : 4-step sequential observation
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

        return action, q_now

    # 学習系メソッド
    def agent_step_update(self, reward, time, action, q_now):
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

        print('Action:%d  Reward:%.1f Q_max:%3f' % (
            self.q_net.action_to_index(action), reward, q_max))

        self.last_action = copy.deepcopy(action)
        self.last_state = self.state.copy()
        print "Record last State and Action"

        # save model
        if np.mod(time,self.q_net.save_model_freq) == 0:
            print "-------------------Save Model-------------------"
            self.q_net.save_model(self.folder,time)

    def check_death(self,image):
        # image -> obs_array
        obs_array = self.feature_extractor.feature(image)
        self.state = np.asanyarray([obs_array], dtype=np.uint8)
        # obs_array -> NN -> get Q_MAX
        self.state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            self.state_ = cuda.to_gpu(self.state_)

        # Generate an Action by e-greedy action selection
        action,q = self.q_net_sim.e_greedy(self.state_, 0)
        q_max = q.ravel()[action]

        self.q_max_max = max(q_max,self.q_max_max)
        death_value = self.q_max_max/2.0

        print "death_value : %3f"%(death_value)
        print "Model_Sim Q_MAX:%3f"%(q_max)
        death = q_max < death_value
        print "Chack Death : %r"%(death)
        return death, action, q
