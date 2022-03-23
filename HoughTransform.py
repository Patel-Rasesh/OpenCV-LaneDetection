from cmath import inf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import time
from ToMergeLines import *
from cv2 import imshow

start = time.time()
# 1. Read an image
laneOriginal = cv2.imread("c:/Users/Friday/Desktop/Spring22/CS6384/Projects/Project1/lane/lane1.jpeg")
cv2.namedWindow('TestImageofLane', cv2.WINDOW_AUTOSIZE)

# 2. Convert BGR to HSV
RGBLaneOriginal = cv2.cvtColor(laneOriginal, cv2.COLOR_BGR2RGB)
hsvLaneOriginal = cv2.cvtColor(laneOriginal, cv2.COLOR_BGR2HSV)

# 3. This mask looks for tarmac gray
# Temporarily initiating with hardcoded values for each lane image
lower_gray = np.array([22,24,26])
upper_gray = np.array([113,104,100])
RGBLaneBlurred = cv2.GaussianBlur(RGBLaneOriginal, (5,5), 0)
maskTarmac = cv2.inRange(RGBLaneBlurred, lower_gray, upper_gray)

# 4. Performing Dilation
kernelTarmac = np.ones((3,3),np.uint8)
dilatedTarmac = cv2.dilate(maskTarmac, kernelTarmac, iterations=1)
'''
# 5. To detect lane lines
# This mask looks for white stripes
lower_white = np.array([230,230,230])
upper_white = np.array([255,255,255])
maskWhiteStripes = cv2.inRange(RGBLaneBlurred, lower_white, upper_white)
# This mask looks for yellow stripes
lower_yellow = np.array([0,230,230])
upper_yellow = np.array([0,255,255])
maskYellowStripes = cv2.inRange(RGBLaneBlurred, lower_yellow, upper_yellow)
# 5. Apply thresholding on maskTarmac
maskYW = cv2.bitwise_or(maskWhiteStripes, maskYellowStripes, dilatedTarmac)
'''
ret, threshTaramc = cv2.threshold(dilatedTarmac,100,255,cv2.THRESH_BINARY)

# 6. Keep only portion from the original image that appear in mask
#tarmacApplied = cv2.subtract(grayLaneTest, dilatedTarmac)
print(RGBLaneOriginal.shape)
RGBLaneSlave = RGBLaneOriginal.copy()

# Temporarily considering only half of the original image
def splittingImage(RGBLaneSlave):
    for x in range(RGBLaneSlave.shape[0]*1//2):
        for y in range(RGBLaneSlave.shape[1]):
            for z in range(RGBLaneSlave.shape[2]):
                RGBLaneSlave[x][y][z] = 0
    return RGBLaneSlave

def customeROI(RGBLaneSlave, threshTaramc):

   # Returns respective Region Of Interest

    for row in range(RGBLaneSlave.shape[0]):
        for col in range(RGBLaneSlave.shape[1]):
            for color in range(RGBLaneSlave.shape[2]):
                if threshTaramc[row][col] == 0:
                    RGBLaneSlave[row][col][color] = 0
    return RGBLaneSlave

# After applying cutome ROI (Detect gray), halving the image
RGBLaneSlave = splittingImage(RGBLaneSlave)
RGBLaneSlave = customeROI(RGBLaneSlave, threshTaramc)

hsvSlaveMask = cv2.cvtColor(RGBLaneSlave, cv2.COLOR_BGR2HSV)
graySlaveMask = cv2.cvtColor(RGBLaneSlave, cv2.COLOR_BGR2GRAY)

# Detecting and drawing line using Hough Transform
graySlaveMaskBlurred = cv2.GaussianBlur(graySlaveMask, (5,5), 0)
edgesSlave = cv2.Canny(graySlaveMaskBlurred, 50, 200, apertureSize=3)
kernelEdges = np.ones((11,11),np.uint8)
#dilatedEdges = cv2.dilate(edgesSlave, kernelEdges, iterations=3)
minLineLength = 100
maxLineGap = 5
lines = cv2.HoughLinesP(edgesSlave,cv2.HOUGH_PROBABILISTIC, np.pi/180, 50, minLineLength,maxLineGap)
print("Number of lines found in the image : ", len(lines))
mergedLines = process_lines(lines, edgesSlave)
print("Number of lines after processing in the image : ", len(mergedLines))

def computeSlope(x1, x2, y1, y2):
    slope = (y2-y1)/(x2-x1)
    if slope == 0 or slope == inf:
        return False
    else:
        return True
#for line in lines:
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        pts = np.array([[x1, y1], [x2 , y2]], np.int32)
        #pts = np.array([[line[0][0], line[1][0]], [line[0][1], line[1][1]]], np.int32)
        if computeSlope(x1, x2, y1, y2):
            cv2.polylines(RGBLaneOriginal, [pts], True, (255,0,0), thickness=10)

plt.subplot(1,2,1)
plt.imshow(RGBLaneOriginal)
plt.subplot(1,2,2)
plt.imshow(RGBLaneSlave)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

end = time.time()
print("Runtime = ", end-start)