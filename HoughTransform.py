import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from ToMergeLines import *
#from KMeans import *

class HoughTransform:
    def craftingMask(self, path):
        '''
        Pre-processing required in setting up the images and the mask
        '''
        # 1. Read an image
        originalImage = cv2.imread(path)
        cv2.namedWindow('TestImageofLane', cv2.WINDOW_AUTOSIZE)
        
        # 2. Convert BGR to RGB for easier interpretation
        RGBimg = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
        #hsvLaneOriginal = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

        # 3. Resizing the image to a fixed resolution
        RGBimgCopy = RGBimg.copy()
        resizedImage = self.resize(RGBimgCopy)
        print("Image has been resized to : ", resizedImage.shape)

        # 4. Applying an elliptical ROI to the image
        ellipseCopy1 = resizedImage.copy()
        resizedImageCopy = resizedImage.copy()
        
        self.drawEllipse(ellipseCopy1)
        self.ellipseROI(resizedImage, ellipseCopy1)

        # 5. This mask looks for tarmac gray
        # Temporarily initiating with hardcoded values for each lane image
        # The following is in RGB order
        lower_gray = np.array([116,114,124])
        upper_gray = np.array([176,170,173])
        RGBBlurred = cv2.GaussianBlur(resizedImage, (15,15), 0)
        maskTarmac = cv2.inRange(RGBBlurred, lower_gray, upper_gray)

        # This mask looks for yellow stripes
        lower_yellow = np.array([180,160,50])
        upper_yellow = np.array([220,200,120])
        maskYellowStripes = cv2.inRange(RGBBlurred, lower_yellow, upper_yellow)
        #maskYellowStripes = cv2.inRange(resizedImage, lower_yellow, upper_yellow)

        # 6. Performing Dilation
        #kernelTarmac = np.ones((3,3),np.uint8)
        kernel_dilation = np.ones((3,3),np.uint8)
        #erodedTarmac = cv2.erode(maskTarmac,kernelTarmac,iterations = 1)
        dilatedROI = cv2.dilate(maskTarmac+maskYellowStripes,kernel_dilation,iterations = 1)

        # 7. Applying threshold
        ret, threshTaramc = cv2.threshold(dilatedROI,100,255,cv2.THRESH_BINARY)

        return resizedImageCopy, resizedImage, threshTaramc

    # WE ARE NO LONGER USING THIS FUNCTION
    def splittingImage(self, img):
        # Splitting the image in half (horizontally) and only considering the bottom part
        for x in range(img.shape[0]*1//2):
            for y in range(img.shape[1]):
                for z in range(img.shape[2]):
                    img[x][y][z] = 0
        return img

    # WE ARE NO LONGER USING THIS FUNCTION
    def customeROI(self, img, mask):
        # Returns respective Region Of Interest
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                for color in range(img.shape[2]):
                    if mask[row][col] == 0:
                        img[row][col][color] = 0
        return img
    
    def ellipseROI(self, img, mask):
        # Returns respective Region Of Interest in elliptical shape
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                for color in range(img.shape[2]):
                    if mask[row][col][color] != 0:
                        img[row][col][color] = 0
        return img

    def houghTransformHandler(self, img):
        '''
        Detects lines based on Hough Parametric transform
        input : image
        returns : list of lines
        '''
        # 1. Converting the image into grayscale
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Detecting and drawing line using Hough Transform
        grayBlurredImage = cv2.GaussianBlur(grayImage, (5,5), 0)
        edgesCanny = cv2.Canny(grayBlurredImage, 50, 200, apertureSize=3)
        
        #kernelEdges = np.ones((11,11),np.uint8)
        #dilatedEdges = cv2.dilate(edgesCanny, kernelEdges, iterations=3)

        # 3. Finding lines in the image using Hough probabilistic approach
        minLineLength = 5
        maxLineGap = 10
        unprocessedLines = cv2.HoughLinesP(edgesCanny,cv2.HOUGH_PROBABILISTIC, np.pi/180, 50, minLineLength,maxLineGap)
        print("Number of lines originally found : ", len(unprocessedLines))
        processedLines = process_lines(unprocessedLines, edgesCanny)
        print("Number of lines after processing : ", len(processedLines))
        
        return processedLines, edgesCanny

    def computeSlope(self, x1, x2, y1, y2):
        '''
        Computes slope and filters lines with reasonable slope
        '''
        if x2-x1 == 0:
            return False
        slope = (y2-y1)/(x2-x1)
        #TODO - Frame logic for slope between 45 to -45
        if slope == 0:
            return False
        else:
            return True

    # WE ARE NO LONGER USING THIS FUNCTION
    def computeThickness(self, img):
        '''
        Decides the thickness of lines based on the resolution of the image
        '''
        thickness = 0
        resolution = img.shape[0]*img.shape[1]
        if resolution < 500000:
            thickness = 2
        elif resolution < 1000000:
            thickness = 7
        else:
            thickness = 10
        return thickness

    def drawLines(self, lines, img):
        '''
        Uses polylines to draw lines on the image
        '''
        thickness = self.computeThickness(img)
        for x in lines:
            #for x1,y1,x2,y2 in x:
            #for [x1,y1],[x2,y2] in x:
            [x1,y1],[x2,y2] = x
            pts = np.array([[x1, y1], [x2 , y2]], np.int32)
            if self.computeSlope(x1, x2, y1, y2):
                cv2.polylines(img, [pts], True, (255,0,0), thickness)

    def resize(self, img):
        '''
        Takes the original image and resize it by the factor of 60%
        '''
        print("Dimensions of the original image = ", img.shape)

        # To keep EVERY image 720x1020, we are scaling dynamically
        scale_percent_width = 1080/img.shape[1]
        scale_percent_height = 720/img.shape[0]
        width = int(img.shape[1] * scale_percent_width)
        height = int(img.shape[0] * scale_percent_height)
        dim = (width, height)
        
        # Resize image
        resizedImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resizedImg 

    def drawEllipse(self, img):
        '''
        Drawing ellipse at the predetermined location of the image
        '''
        # Starts from bottom center
        center_coordinates = (540, 720)
        axesLength = (700, 400)
        angle = 0
        startAngle = 0
        endAngle = 360
        color = (0, 0, 0)
        thickness = -1
        img = cv2.ellipse(img, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
        return img

    def driver(self, path):
        '''
        Main driving function
        '''
        start = time.time()
        # 1. Preprocessing on the image
        RGBOriginal, ROIImage, threshTaramc = self.craftingMask(path)

        ROIImageCopy = ROIImage.copy()
        # ROIImageCopy = self.customeROI(ROIImageCopy, threshTaramc)

        # 2. Applying Hough Transform
        processedLines, edgesCanny = self.houghTransformHandler(ROIImageCopy)

        # 3. Drawing detected lines on top of the original input image
        self.drawLines(processedLines, RGBOriginal)

        # 4. Plotting original image with edgesCanny from Canny 
        plt.subplot(1,2,1)
        plt.imshow(RGBOriginal)
        plt.subplot(1,2,2)
        plt.imshow(edgesCanny)
        end = time.time()
        plt.show()
        print("Runtime = ", end-start)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

obj = HoughTransform()
path = "c:/Users/Friday/Desktop/Spring22/CS6384/Projects/Project1/lane/lane1.jpeg"
obj.driver(path)
