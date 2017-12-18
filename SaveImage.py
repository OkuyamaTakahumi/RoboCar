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
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.gray()


def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def to_grayscale(img):
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayed

def pause_Image_plot(img1,img2):
    ax1.cla()
    ax1.imshow(img1)
    ax2.cla()
    ax2.imshow(img2)

if __name__ == '__main__':
    folder = args.folder


    context1 = zmq.Context()
    socket_local = context1.socket(zmq.REP)
    socket_local.bind("tcp://*:5555")


    print "Waiting Request..."

    i = 0
    while True:
        # Receve Data from C++ Program
        #data =  socket_local.recv()

        # Receve Data from C++ Program
        data1, data2=  socket_local.recv_multipart()

        print "Received RoboCar's original Image"
        image1 = np.frombuffer(data1, dtype=np.uint8);
        image1 = image1.reshape((544,960,3))

        image2 = np.frombuffer(data2, dtype=np.uint8);
        image2 = image2.reshape((600,800,3))

        cv2.imwrite("./Images1/%d.png"%(i),image1)
        cv2.imwrite("./Images2/%d.png"%(i),image2)
        print "Save Image!!!"
        print "PhotoNumber : %d"%(i)

        if(args.image):
            pause_Image_plot(to_plot(image1),to_plot(image2))
            plt.pause(1.0 / 10**10) #引数はsleep時間

        #  Send reply back to client
        socket_local.send(np.array([1]))

        print "---------------------------------------------------------------"

        i+=1
