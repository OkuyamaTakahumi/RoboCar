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

from Agent import CnnDqnAgent
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

def send_action(action,receive_time):
    socket.send(np.array([action])) #  Send reply back to client
    run_mes = (time.time() - receive_time)*1000
    print "Send Action : %d"%(action)

if __name__ == '__main__':
    gpu = args.gpu
    use_adati = args.adati
    folder = args.folder
    model_num = args.model_num
    NN = args.NN
    test = args.test
    log_file = args.log_file
    image_num = args.image
    plot_q_value = args.q_value

    death = True
    cycle_counter = 0
    episode_num = 0

    if(use_adati):
        print "Use Adati's NN"
        img_pro = ImageProcessing_adati(image_num,plot_q_value)
    else:
        img_pro = ImageProcessing(image_num,plot_q_value)

    a_num = 3
    if(folder=="ModelAction3V_Real"):
        a_num = 5

    if(NN):
        agent = CnnDqnAgent();
        agent.agent_init(
            use_gpu = gpu,
            folder = folder,
            model_num = model_num,
            a_num = a_num,
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
        receive_time = time.time()
        image = np.frombuffer(data, dtype=np.uint8);
        image = image.reshape((227,227,3))

        new_image_g = img_pro.lane_detection(image)
        new_image = cv2.merge((new_image_g,new_image_g,new_image_g))

        if(NN):
            q_now = np.zeros((a_num))
            if(death):
                death = img_pro.check_death(new_image_g,revived=True) #1chanel

                if(death):
                    print "Agent is Death"
                    action = 100
                else:
                    action, q_now = agent.agent_start(new_image)

                    episode_num += 1
                    print "Episode %d START"%(episode_num)
                    #episode_start_time = cycle_counter
                    cycle_counter += 1

                    reward = 0.05
                    score = 0

                send_action(action,receive_time)

            else:
                death = img_pro.check_death(new_image_g,revived=False) #1chanel

                if(death):
                    action = 100 # back
                    if(not test):
                        #ep_cycle_num = cycle_counter - episode_start_time

                        print "Finish Episode%d"%(episode_num)
                        print "Score is %.3f"%(score)

                        action = 101 #stop
                        # logファイルへの書き込み
                        with open(log_file, 'a') as the_file:
                            the_file.write(str(cycle_counter)+','+str(score) +','+str(episode_num) + '\n')

                        send_action(action,receive_time)
                        agent.agent_end(-1,cycle_counter)

                    else:
                        send_action(action,receive_time)

                else:
                    action, q_now = agent.agent_step(new_image)

                    send_action(action,receive_time)
                    if(not test):
                        agent.agent_step_update(reward,cycle_counter,action,q_now)

                        score += reward

                        if(action ==1):
                            reward = 0.1
                        elif(action == 3 or action ==4):
                            reward = 0.075
                        else:
                            reward = 0.05
                    cycle_counter += 1
        else:
            death = img_pro.check_death(new_image_g,revived=False)
            print "Death=",death
            q_now = np.random.rand(a_num)
            action = 1
            if(cycle_counter%9 < 3):
                action = 0
            elif(cycle_counter%9 < 6):
                action = 1
            else:
                action = 2
            send_action(action,receive_time)
            cycle_counter += 1
            # img_pro.save_image(img=new_image_g,photo_id=cycle_counter+400,dir_path='./SaveImageStraight/')

        #title = np.sum(new_image_g[:150,:]/255.0)
        #hoge1,hoge2,hoge3 = img_pro.change_speed(new_image_g)
        #title = 'right:%d, left:%d, divide:%.3f' % (hoge1,hoge2,hoge3)
        #img_pro.plot(new_image_g, new_image, q_now ,title=title, a_num=a_num)

        img_pro.plot(new_image_g, new_image, q_now , a_num=a_num)
        #img_pro.plot(img_pro.to_plot(image), new_image_g, q_now ,a_num=a_num)

        #detect_image = img_pro.to_plot(img_pro.make_detection_image(image,new_image_g))
        #img_pro.plot(detect_image, new_image, q_now, a_num=a_num)

        print "---------------------------------------------------------"
        print "Step : %d"%(cycle_counter)
