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

    action_log_data_size = 30
    action_log = np.empty(action_log_data_size, dtype=int)
    action_log[:] = np.nan
    action_stock_num = 0
    cycle_of_episode = 0

    def print_action_log_data(self):
        print "cycle_of_episode : ",self.cycle_of_episode
        print self.action_log
        N0 =  len(np.where(self.action_log ==0)[0]) # 0の数を取得
        N2 =  len(np.where(self.action_log ==2)[0]) # 2の数を取得
        print u"直近の各アクションの数[Left,Right]:",[N0,N2]

    def stock_action_log(self,action):
        if(action == 0 or action == 2):
            data_index = self.action_stock_num % self.action_log_data_size
            self.action_log[data_index] = action
            self.action_stock_num += 1

    def get_initial_action(self):
        if(len(np.where(self.action_log ==0)[0]) > len(np.where(self.action_log ==2)[0])):
            return 0
        else:
            return 2

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

        else: # 初めてLISを起動する時
            self.feature_extractor = CnnFeatureExtractor(self.use_gpu, self.model, self.model_type, self.q_net_input_dim)
            pickle.dump(self.feature_extractor, open(self.cnn_feature_extractor, 'w'))
            print("pickle.dump finished")

        self.q_net = QNet(self.use_gpu,self.actions,self.q_net_input_dim)
        self.q_net.load_model(self.folder,model_num)

    # 行動取得系,state更新系メソッド
    def agent_start(self, image):
        try:
            obs_array = self.feature_extractor.feature(image)
            # Initialize State
            self.state = np.zeros((self.q_net.hist_size, self.q_net_input_dim), dtype=np.uint8)
            self.state[0] = obs_array

            # Generate an Action e-greedy
            self.initial_action = self.get_initial_action()
            q_now = np.zeros((len(self.actions)))
            return_action = self.initial_action

            # Update for next step
            self.last_action = copy.deepcopy(return_action)
            self.last_state = self.state.copy()

            self.cycle_of_episode = 0
            self.print_action_log_data()

            return return_action, q_now
        except:
            import traceback
            import sys
            traceback.print_exc()
            sys.exit()

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
        try:
            obs_array = self.feature_extractor.feature(image)
            if self.q_net.hist_size == 4:
                self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_array], dtype=np.uint8)
            elif self.q_net.hist_size == 2:
                self.state = np.asanyarray([self.state[1], obs_array], dtype=np.uint8)
            elif self.q_net.hist_size == 1:
                self.state = np.asanyarray([obs_array], dtype=np.uint8)
            else:
                print("self.DQN.hist_size err")

            self.cycle_of_episode += 1

            self.print_action_log_data()
            if(self.cycle_of_episode<=5):
                action = self.initial_action
                q_now = np.zeros((len(self.actions)))
                return action, q_now

            state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
            if self.use_gpu >= 0:
                state_ = cuda.to_gpu(state_)

            action, q_now = self.q_net.e_greedy(state_, 0)

            self.stock_action_log(action)

            return action, q_now
        except:
            import traceback
            import sys
            traceback.print_exc()
            sys.exit()

    # 学習系メソッド
    def agent_step_update(self, reward, time, action, q_now):
        try:
            # Learning Phase
            self.q_net.stock_experience(time, self.last_state, self.last_action, reward, self.state, False)
            self.q_net.experience_replay(time)

            # Target model update
            if np.mod(time, self.q_net.target_model_update_freq) == 0:
                print("Model Updated")
                self.q_net.target_model_update()

            # Simple text based visualization
            if(self.use_gpu >= 0 and self.cycle_of_episode > 5):
                q_max = np.max(q_now.get())
            else:
                q_max = np.max(q_now)

            print('Action:%d  Reward:%.3f Q_max:%3f' % (
                self.q_net.action_to_index(action), reward, q_max))

            self.last_action = copy.deepcopy(action)
            self.last_state = self.state.copy()

            # save model
            if np.mod(time,self.q_net.save_model_freq) == 0:
                print "-------------------Save Model-------------------"
                self.q_net.save_model(self.folder,time)
        except:
            import traceback
            import sys
            traceback.print_exc()
            sys.exit()
