# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import numpy as np
import os
print 3
plt.gray()


def to_plot(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    global cache
    global first_frame
    y_global_min = img.shape[0] #min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [],[]
    l_lane,r_lane = [],[]
    det_slope = 0.4
    alpha =0.2
    #i got this alpha value off of the forums for the weighting between frames.
    #i understand what it does, but i dont understand where it comes from
    #much like some of the parameters in the hough function

    for line in lines:
        #1
        for x1,y1,x2,y2 in line:
            slope = get_slope(x1,y1,x2,y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        #2
        y_global_min = min(y1,y2,y_global_min)

    # to prevent errors in challenge video from dividing by zero
    if((len(l_lane) == 0) or (len(r_lane) == 0)):
        print ('no lane detected')
        return 1

    #3
    l_slope_mean = np.mean(l_slope,axis =0)
    r_slope_mean = np.mean(r_slope,axis =0)
    l_mean = np.mean(np.array(l_lane),axis=0)
    r_mean = np.mean(np.array(r_lane),axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0 )):
        print('dividing by zero')
        return 1



    #4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    #5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b)/l_slope_mean)
    l_x2 = int((y_max - l_b)/l_slope_mean)
    r_x1 = int((y_global_min - r_b)/r_slope_mean)
    r_x2 = int((y_max - r_b)/r_slope_mean)

    #6
    if l_x1 > r_x1:
        l_x1 = int((l_x1+r_x1)/2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1 ) + l_intercept)
        r_y1 = int((r_slope_mean * r_x1 ) + r_intercept)
        l_y2 = int((l_slope_mean * l_x2 ) + l_intercept)
        r_y2 = int((r_slope_mean * r_x2 ) + r_intercept)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1,l_y1,l_x2,l_y2,r_x1,r_y1,r_x2,r_y2],dtype ="float32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else :
        prev_frame = cache
        next_frame = (1-alpha)*prev_frame+alpha*current_frame

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]),int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]),int(next_frame[7])), color, thickness)

    cache = next_frame

def get_slope(x1,y1,x2,y2):
    return (y2-y1)/(x2-x1)

imgDir_path = '/Users/okuyamatakashi/RoboCar/RoboCarImage/'
imgDir_path = 'C:\Users\Student2015\Desktop/RoboCar/RoboCarImage/'

file_list = os.listdir(imgDir_path)
for File in file_list:
    if '.jpg' not in File and '.jpeg' not in File and '.png' not in File and '.JPG' not in File:
        file_list.remove(File)
image = cv2.imread(imgDir_path+file_list[224])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#src画像から指定した色の範囲にもとづいてマスク画像を生成する
mask_white = cv2.inRange(gray_image, 200, 255)
plt.imshow(mask_white)

#mask_image_not = cv2.bitwise_not(gray_image, mask_white)
#mask_image_or = cv2.bitwise_or(gray_image, mask_white)
mask_image = cv2.bitwise_and(gray_image, mask_white)
mask_image.shape
plt.imshow(mask_image)
# 平滑化
kernel_size = 5
gauss_gray = cv2.GaussianBlur(mask_image, (kernel_size, kernel_size), 0)
plt.imshow(gauss_gray)
# エッジ検出
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gauss_gray, low_threshold, high_threshold)
canny_edges.shape
plt.imshow(canny_edges)

# roi -> Region Of Interest
imgshape = image.shape
print imgshape
lower_left = [imgshape[1]/9,imgshape[0]]
lower_right = [imgshape[1]-imgshape[1]/9,imgshape[0]]
top_left = [imgshape[1]/2-imgshape[1]/8,imgshape[0]/2+imgshape[0]/10]
top_right = [imgshape[1]/2+imgshape[1]/8,imgshape[0]/2+imgshape[0]/10]

lower_left = [0,100]
lower_right = [227,100]
top_left = [75,0]
top_right = [227-75,0]
vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]

print lower_left
print lower_right
print top_left
print top_right
print vertices[0].shape
#defining a blank mask to start with
roi_mask = np.zeros_like(mask_white)

#filling pixels inside the polygon defined by "vertices" with the fill color
cv2.fillPoly(roi_mask, vertices, 255)
plt.imshow(roi_mask)
roi_image = mask_white.copy()
roi_image = cv2.bitwise_and(mask_white, roi_mask)
plt.imshow(roi_image)

#rho and theta are the distance and angular resolution of the grid in Hough space
#same values as quiz
rho = 4
theta = np.pi/180
#threshold is minimum number of intersections in a grid for candidate line to go to output
threshold = 30
min_line_len = 100
max_line_gap = 180
#my hough values started closer to the values in the quiz, but got bumped up considerably for the challenge video
lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
lines

plt.imshow(to_plot(image))
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)

plt.imshow(to_plot(image))

line_img = np.zeros((canny_edges.shape[0], canny_edges.shape[1], 3), dtype=np.uint8)


line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)

draw_lines(line_img,lines)
