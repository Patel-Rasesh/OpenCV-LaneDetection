import cv2
from cv2 import imshow
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import time

start = time.time()
# 1. Read an image
laneTest = cv2.imread("c:/Users/Friday/Desktop/Spring22/CS6384/Projects/Project1/lane/lane1.jpeg")
cv2.namedWindow('TestImageofLane', cv2.WINDOW_AUTOSIZE)

# Convert BGR to HSV
RGBLaneTest = cv2.cvtColor(laneTest, cv2.COLOR_BGR2RGB)
hsvLaneTest = cv2.cvtColor(laneTest, cv2.COLOR_BGR2HSV)

# Optimize - use Mouse click event and confirm the threshold values
# This mask looks for tarmac gray
lower_gray = np.array([75,75,60])
upper_gray = np.array([135,126,121])
RGBLaneTestBlurred = cv2.GaussianBlur(RGBLaneTest, (5,5), 0)
maskTarmac = cv2.inRange(RGBLaneTestBlurred, lower_gray, upper_gray)

# Performing Dilation
kernel = np.ones((3,3),np.uint8)
dilatedTarmac = cv2.dilate(maskTarmac,kernel,iterations = 2)
# Apply thresholding on maskTarmac
ret,thresh = cv2.threshold(dilatedTarmac,100,255,cv2.THRESH_BINARY)

# Keep only portion from the original image that appear in mask
#tarmacApplied = cv2.subtract(grayLaneTest, dilatedTarmac)
print(RGBLaneTest.shape)
RGBLaneTestSlave = RGBLaneTest.copy()

for a in range(RGBLaneTestSlave.shape[0]):
    for b in range(RGBLaneTestSlave.shape[1]):
        for c in range(RGBLaneTestSlave.shape[2]):
            if thresh[a][b] == 0:
                RGBLaneTestSlave[a][b][c] = 0

hsvLaneTestMask = cv2.cvtColor(RGBLaneTestSlave, cv2.COLOR_BGR2HSV)
# To detect lane lines
# This mask looks for white stripes
lower_white = np.array([0,0,200])
upper_white = np.array([145,60,255])
maskWhiteStripes = cv2.inRange(hsvLaneTestMask, lower_white, upper_white)
# This mask looks for yellow stripes
lower_yellow = np.array([22,93,0])
upper_yellow = np.array([45,255,255])
maskYellowStripes = cv2.inRange(hsvLaneTestMask, lower_yellow, upper_yellow)

maskYW = cv2.bitwise_or(maskWhiteStripes, maskYellowStripes)

#targetImage = cv2.bitwise_and(RGBLaneTest, maskYW)
grayLaneTestMask = cv2.cvtColor(RGBLaneTestSlave, cv2.COLOR_BGR2GRAY)

# For line detection
grayLaneTestMaskBlurred = cv2.GaussianBlur(grayLaneTestMask, (5,5), 0)
edges = cv2.Canny(grayLaneTestMaskBlurred,50,200,apertureSize = 3)
minLineLength = 100
maxLineGap = 50
lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 100, minLineLength,maxLineGap)
print("Number of lines found in the image : ",len(lines))
# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
#         #cv2.line(RGBLaneTest,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
#         pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
#         cv2.polylines(RGBLaneTest, [pts], True, (0,255,0), thickness=7)

def display_lines(image, lines):
    #lines_image = np.zeros_like(image)
    #make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            #draw lines on a black image
            #cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
            cv2.polylines(image, [pts], True, (0,255,0), thickness=7)     
    return image

def average(image, lines):
    left = []
    right = []

    if lines is not None:
      for line in lines:
        #print(line)
        x1, y1, x2, y2 = line.reshape(4)
        #fit line to points, return slope and y-int
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        #print(parameters)
        slope = parameters[0]
        y_int = parameters[1]
        #lines on the right have positive slope, and lines on the left have neg slope
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))
            
    #takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    #create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])

def make_points(image, average):
    #print(average)
    slope, y_int = average
    y1 = image.shape[0]
    #how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (2/5))
    #determine algebraically
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

averaged_lines = average(RGBLaneTestSlave, lines)
black_lines = display_lines(RGBLaneTest, averaged_lines)

plt.subplot(1,2,1)
plt.imshow(RGBLaneTestSlave)
plt.subplot(1,2,2)
plt.imshow(black_lines)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

end = time.time()
print("Runtime = ", end-start)