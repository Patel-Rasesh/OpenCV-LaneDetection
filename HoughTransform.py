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
        laneOriginal = cv2.imread(path)
        cv2.namedWindow('TestImageofLane', cv2.WINDOW_AUTOSIZE)
        
        # 2. Convert BGR to HSV
        RGBLaneOriginal = cv2.cvtColor(laneOriginal, cv2.COLOR_BGR2RGB)
        hsvLaneOriginal = cv2.cvtColor(laneOriginal, cv2.COLOR_BGR2HSV)

        RGBLaneSlave = RGBLaneOriginal.copy()
        slaveResized = self.resize(RGBLaneSlave)
        print(slaveResized.shape)
        RGBSlaveEllipse = slaveResized.copy()
        self.drawEllipse(RGBSlaveEllipse)
        ellipseSlave = self.ellipseROI(slaveResized, RGBSlaveEllipse)

        # 3. This mask looks for tarmac gray
        # Temporarily initiating with hardcoded values for each lane image
        # The following is in RGB order
        lower_gray = np.array([116,114,124])
        upper_gray = np.array([176,170,173])
        RGBLaneBlurred = cv2.GaussianBlur(ellipseSlave, (15,15), 0)
        maskTarmac = cv2.inRange(RGBLaneBlurred, lower_gray, upper_gray)

        # This mask looks for yellow stripes
        lower_yellow = np.array([180,160,50])
        upper_yellow = np.array([220,200,120])
        maskYellowStripes = cv2.inRange(ellipseSlave, lower_yellow, upper_yellow)

        # 4. Performing Dilation
        kernelTarmac = np.ones((3,3),np.uint8)
        kernel_lg = np.ones((3,3),np.uint8)
        # image processing technique called the erosion is used for noise reduction
        erodedTarmac = cv2.erode(maskTarmac,kernelTarmac,iterations = 1)
        # image processing technique called the dilation is used to regain some lost area
        dilatedTarmac = cv2.dilate(maskTarmac+maskYellowStripes,kernel_lg,iterations = 1)
        # Bitwise-AND to get black everywhere else except the region of interest
        #result = cv2.bitwise_and(frame,frame, mask= mask)
        #dilatedTarmac = cv2.dilate(maskTarmac, kernelTarmac, iterations=1)
        '''
        # 5. To detect lane lines
        # This mask looks for white stripes
        lower_white = np.array([230,230,230])
        upper_white = np.array([255,255,255])
        maskWhiteStripes = cv2.inRange(RGBLaneBlurred, lower_white, upper_white)
        # 5. Apply thresholding on maskTarmac
        maskYW = cv2.bitwise_or(maskWhiteStripes, maskYellowStripes, dilatedTarmac)
        '''
        ret, threshTaramc = cv2.threshold(dilatedTarmac,100,255,cv2.THRESH_BINARY)

        return slaveResized, ellipseSlave, threshTaramc
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
    
    def ellipseROI(self, img, mask):

    # Returns respective Region Of Interest
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                for color in range(img.shape[2]):
                    if mask[row][col][color] != 0:
                        img[row][col][color] = 0
        return img

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
        
        return mergedLines, edgesSlave

    def computeSlope(self, x1, x2, y1, y2):
        if x2-x1 == 0:
            return False
        slope = (y2-y1)/(x2-x1)
        #TODO - Even the slope=1 shouldn't be detected
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
    def resize(self, img):
        '''
        Takes the original image and resize it by the factor of 60%
        '''
        print("Dimensions of the original image = ", img.shape)
        scale_percent_width = 1080/img.shape[1]
        scale_percent_height = 720/img.shape[0]
        width = int(img.shape[1] * scale_percent_width)
        height = int(img.shape[0] * scale_percent_height)
        dim = (width, height)
        
        # resize image
        resizedImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ',resizedImg.shape)
        return resizedImg 

    def drawEllipse(self, image):
        center_coordinates = (540, 720)
        axesLength = (700, 400)
        angle = 0
        startAngle = 0
        endAngle = 360
        color = (0, 0, 0)
        thickness = -1
        image = cv2.ellipse(image, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)
        return image
    def driver(self, path):
        start = time.time()
        RGBLaneOriginal, ROIApplied, threshTaramc = self.craftingMask(path)
        # After applying cutome ROI (Detect gray), halving the image
        RGBLaneSlave = ROIApplied.copy()
        # RGBLaneSlave = self.customeROI(RGBLaneSlave, threshTaramc)
        mergedLines, edges = self.houghTransformHandler(RGBLaneSlave)
        self.drawLines(mergedLines, RGBLaneOriginal)
        plt.subplot(1,2,1)
        plt.imshow(RGBLaneOriginal)
        plt.subplot(1,2,2)
        plt.imshow(edges)
        end = time.time()
        plt.show()
        print("Runtime = ", end-start)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

obj = HoughTransform()
path = "c:/Users/Friday/Desktop/Spring22/CS6384/Projects/Project1/lane/lane4.jpeg"
obj.driver(path)
