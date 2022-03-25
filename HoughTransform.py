import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys, os
import time
from ToMergeLines import *
#from KMeans import *


class HoughTransform:
    def craftingMask(self, path):
        '''
        Pre-processing required in setting up the images and the mask
        '''
        # 1. Read an image
        laneOriginal = cv2.imread(path)
        cv2.namedWindow('TestImageofLane', cv2.WINDOW_AUTOSIZE)
        
        # 2. Convert BGR to HSV
        RGBLaneOriginal = cv2.cvtColor(laneOriginal, cv2.COLOR_BGR2RGB)
        hsvLaneOriginal = cv2.cvtColor(laneOriginal, cv2.COLOR_BGR2HSV)

        # 3. This mask looks for tarmac gray
        # Temporarily initiating with hardcoded values for each lane image
        lower_gray = np.array([60,75,75])
        upper_gray = np.array([130,126,135])
        RGBLaneBlurred = cv2.GaussianBlur(RGBLaneOriginal, (15,15), 0)
        maskTarmac = cv2.inRange(RGBLaneBlurred, lower_gray, upper_gray)

        # 4. Performing Dilation
        kernelTarmac = np.ones((3,3),np.uint8)
        kernel_lg = np.ones((3,3),np.uint8)
        # image processing technique called the erosion is used for noise reduction
        erodedTarmac = cv2.erode(maskTarmac,kernelTarmac,iterations = 1)
        # image processing technique called the dilation is used to regain some lost area
        dilatedTarmac = cv2.dilate(maskTarmac,kernel_lg,iterations = 1)
        # Bitwise-AND to get black everywhere else except the region of interest
        #result = cv2.bitwise_and(frame,frame, mask= mask)
        #dilatedTarmac = cv2.dilate(maskTarmac, kernelTarmac, iterations=1)
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
        return RGBLaneOriginal, threshTaramc
    # 6. Keep only portion from the original image that appear in mask
    #tarmacApplied = cv2.subtract(grayLaneTest, dilatedTarmac)

    # Temporarily considering only half of the original image
    def splittingImage(self, RGBLaneSlave):
        for x in range(RGBLaneSlave.shape[0]*1//2):
            for y in range(RGBLaneSlave.shape[1]):
                for z in range(RGBLaneSlave.shape[2]):
                    RGBLaneSlave[x][y][z] = 0
        return RGBLaneSlave

    def customeROI(self, RGBLaneSlave, threshTaramc):

    # Returns respective Region Of Interest
        for row in range(RGBLaneSlave.shape[0]):
            for col in range(RGBLaneSlave.shape[1]):
                for color in range(RGBLaneSlave.shape[2]):
                    if threshTaramc[row][col] == 0:
                        RGBLaneSlave[row][col][color] = 0
        return RGBLaneSlave

    def houghTransformHandler(self, RGBLaneSlave):
        '''
        Detects lines based on Hough Parametric transform
        input : image
        returns : list of lines
        '''
        hsvSlaveMask = cv2.cvtColor(RGBLaneSlave, cv2.COLOR_BGR2HSV)
        graySlaveMask = cv2.cvtColor(RGBLaneSlave, cv2.COLOR_BGR2GRAY)

        # Detecting and drawing line using Hough Transform
        graySlaveMaskBlurred = cv2.GaussianBlur(graySlaveMask, (5,5), 0)
        edgesSlave = cv2.Canny(graySlaveMaskBlurred, 50, 200, apertureSize=3)
        #kernelEdges = np.ones((11,11),np.uint8)
        #dilatedEdges = cv2.dilate(edgesSlave, kernelEdges, iterations=3)
        minLineLength = 5
        maxLineGap = 10
        lines = cv2.HoughLinesP(edgesSlave,cv2.HOUGH_PROBABILISTIC, np.pi/180, 50, minLineLength,maxLineGap)
        print("Number of lines found in the image : ", len(lines))
        mergedLines = process_lines(lines, edgesSlave)
        print("Number of lines after processing in the image : ", len(mergedLines))
        
        return mergedLines

    def computeSlope(self, x1, x2, y1, y2):
        if x2-x1 == 0:
            return False
        slope = (y2-y1)/(x2-x1)
        if slope == 0:
            return False
        else:
            return True

    def computeThickness(self, image):
        thickness = 0
        resolution = image.shape[0]*image.shape[1]
        if resolution < 500000:
            thickness = 2
        elif resolution < 1000000:
            thickness = 7
        else:
            thickness = 10
        return thickness

    def drawLines(self, lines, image):
        thickness = self.computeThickness(image)
        for x in lines:
            #for x1,y1,x2,y2 in x:
            #for [x1,y1],[x2,y2] in x:
            [x1,y1],[x2,y2] = x
            pts = np.array([[x1, y1], [x2 , y2]], np.int32)
            if self.computeSlope(x1, x2, y1, y2):
                cv2.polylines(image, [pts], True, (255,0,0), thickness)

    def driver(self, path):
        RGBLaneOriginal, threshTaramc = self.craftingMask(path)
        print(RGBLaneOriginal.shape)
        RGBLaneSlave = RGBLaneOriginal.copy()
        # After applying cutome ROI (Detect gray), halving the image
        RGBLaneSlave = self.splittingImage(RGBLaneSlave)
        RGBLaneSlave = self.customeROI(RGBLaneSlave, threshTaramc)
        self.drawLines(self.houghTransformHandler(RGBLaneSlave), RGBLaneOriginal)
        plt.subplot(1,2,1)
        plt.imshow(RGBLaneOriginal)
        plt.subplot(1,2,2)
        plt.imshow(RGBLaneSlave)
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

obj = HoughTransform()
path = "c:/Users/Friday/Desktop/Spring22/CS6384/Projects/Project1/lane/lane1.jpeg"
start = time.time()
obj.driver(path)
end = time.time()
print("Runtime = ", end-start)