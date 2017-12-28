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
from Adati import ImageProcessing_adati
from image_processing import ImageProcessing

parser = argparse.ArgumentParser(description='ml-agent-for-unity')

parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help=u'GPU ID (negative value indicates CPU)')

parser.add_argument('--folder', '-f', default='ModelRobo3Real', type=str,
                    help=u'モデルの存在するフォルダ名')
parser.add_argument('--model_num', '-m', default=0,type=int,
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

def send_action(action,receive_time):
    #  Send reply back to client
    socket.send(np.array([action]))

    run_mes = (time.time() - receive_time)*1000
    with open("run_time.log", 'a') as the_file:
        the_file.write(str(run_mes) + '\n')
    #print "Send Action : %d"%(action)

if __name__ == '__main__':
    gpu = args.gpu
    use_adati = args.adati
    folder = args.folder
    model_num = args.model_num
    NN = args.NN
    test = args.test
    log_file = args.log_file

    death = True
    cycle_counter = 0
    episode_num = 1
    score = 0



    if(use_adati):
        print "Use Adati's NN"
        img_pro = ImageProcessing_adati(args.image,args.q_value)
    else:
        img_pro = ImageProcessing(args.image,args.q_value)


    a_num = 13
    hidden_dim = 256
    if(folder=="ModelRobo3Real" or folder=="ModelRobo4Real"):
        a_num = 7
    elif(folder=="ModelRobo5_2Real" or folder=="ModelRobo6_2Real" or folder=="ModelRobo6_2_1Real"):
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
        data = socket.recv()
        #print "Received"
        receive_time = time.time()
        image = np.frombuffer(data, dtype=np.uint8);
        image = image.reshape((227,227,3))

        #image = image.reshape((227,227))
        #new_image = cv2.merge((image,image,image))

        if(use_adati):
            detection_image = adati.mainfunction(image)
            new_image_g = cv2.resize(detection_image,(227,227))

        else:
            new_image_g = img_pro.lane_detection(image)
        new_image = cv2.merge((new_image_g,new_image_g,new_image_g))

        if(NN):
            if(death):
                death,test_action,test_q = agent.check_death(new_image)

                if(death):
                    print "Agent is Death"
                    action = 100
                else:
                    if(test):
                        action,q_max = decide_test_action(test_action,test_q,a_num)
                    else:
                        print "Episode %d START"%(episode_num)
                        action = agent.agent_start(new_image)
                    episode_start_time = cycle_counter
                    cycle_counter += 1
                send_action(action,receive_time)

            else:
                death,test_action,test_q = agent.check_death(new_image)

                if(death):
                    action = 100 # back
                    if(not test):
                        ep_score = cycle_counter - episode_start_time
                        print "Episode Score is %d"%(ep_score)
                        score += ep_score
                        if(episode_num%3 == 0):
                            print "Finish %d Episode"%(episode_num)
                            action = 101 #stop
                            # logファイルへの書き込み
                            with open(log_file, 'a') as the_file:
                                the_file.write(str(cycle_counter)+','+str(score) +','+str(episode_num) + '\n')
                            print "Score is %d"%(score)
                            score = 0
                        send_action(action,receive_time)
                        reward = -1
                        agent.agent_end(reward,cycle_counter)
                        episode_num += 1
                    else:
                        send_action(action,receive_time)

                else:
                    if(test):
                        action,q_max = decide_test_action(test_action,test_q,a_num)
                        send_action(action,receive_time)
                    else:
                        action, q_now = agent.agent_step(new_image)
                        send_action(action,receive_time)
                        reward = 0
                        agent.agent_step_update(reward,cycle_counter,action,q_now)
                    cycle_counter += 1
        else:
            test_q = np.random.rand(a_num)
            action = 3
            action,q_max = decide_test_action(action,q,a_num)
            send_action(action,receive_time)

        img_pro.plot(image,new_image,test_q,a_num)

        if(cycle_counter==1000):
            print "1000cycle finish"
            break
        print "---------------------------------------------------------"
        print "Step : %d"%(cycle_counter)
