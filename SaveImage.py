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

parser = argparse.ArgumentParser(description='Images')
parser.add_argument('--folder', '-f', default='Images', type=str,
                    help=u'ImageをSaveするフォルダ名')

parser.add_argument('--image', '-i', action = "store_true",
                    help=u'imageをPlotするか')


args = parser.parse_args()


if(args.image):
    fig, ax1 = plt.subplots(1, 1)
    plt.gray()


def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed

def pause_Image_plot(img):
    ax1.cla()
    ax1.imshow(img)

if __name__ == '__main__':
    folder = args.folder


    context1 = zmq.Context()
    socket_local = context1.socket(zmq.REP)
    socket_local.bind("tcp://*:5555")


    print "Waiting Request..."

    i = 0
    while True:
        # Receve Data from C++ Program
        data =  socket_local.recv()
        print "Received RoboCar's original Image"
        image = np.frombuffer(data, dtype=np.uint8);
        image = image.reshape((600,800,3))
        cv2.imwrite("./Images/%d.png"%(i),image)
        print "Save Image!!!"
        print "PhotoNumber : %d"%(i)

        if(args.image):

            pause_Image_plot(to_plot(image))
            plt.pause(1.0 / 10**10) #引数はsleep時間

        #  Send reply back to client
        socket_local.send(np.array([1]))

        print "---------------------------------------------------------------"

        i+=1
