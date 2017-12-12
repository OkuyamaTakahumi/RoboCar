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
