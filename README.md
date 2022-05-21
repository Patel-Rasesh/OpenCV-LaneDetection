# OpenCV-LaneDetection
This covers lane line detection on images (taken from car dashcam and walkover bridge). Hough transform has been used alongside other image processing operations


# Spot the race-track
Here, the expectation is to be able to detect lane lines from an image- captured from the dash-cam of a car or the camera attached to a foot-over bridge. The configuration for this lane detection is as follows  

• The lane can be straight or a curve
• White broken lane lines should be detected as one merged line along its trajectory
• Two solid parallel lane lines should be counted as two distinct lines
• Fire lanes should be detected
• Yellow lanes should be detected
• An image with no lane lines should not detect any lines
• Detection should not take more than a minute
• Bonus points if able to detect STOP signs

Time spent: **11** hours spent in total

## Steps achieved

- [X] Able to detect broken or solid lane lines
- [X] Able to merge broken white lane lines into single line
- [X] Able to detect fire and yellow lane
- [X] Computation time is less than a minute (on average 15s)

## Screenshot of the Computer Vision pipeline used - 

![image](https://user-images.githubusercontent.com/91232193/169663344-bfc1ed50-5ad5-47a0-9ae0-a3e00b7c33cc.png)

## Screenshots of the execution -  

![image](https://user-images.githubusercontent.com/91232193/169663367-738f4749-914e-4ba1-af76-62a9e4809c23.png)  
![image](https://user-images.githubusercontent.com/91232193/169663373-194c67c6-5dc9-4a4b-a59a-68020e5cba99.png)  
![image](https://user-images.githubusercontent.com/91232193/169663381-b9ff9581-f189-4cc0-bd0f-9e706b0f4063.png)  

## Challenges faced 
(Unresolved challenges are unmarked)

- [X] Numerous trials and errors of morphological and geometric transformation operations before reaching to a satisfactory ROI- Region of Interest, for lane detection
- [X] Calibrating mutually accepted values of parameters i.e. Line Length, Line Gap, that can work for every test image
- [X] Ignoring white horizontal marking on the road
- [X] Struggled while merging nearby lines (within one lane line) detected form Probabilistic Hough Transform (HoughTransformP)
- [ ] Drawing curved lines using Hough Line Transform while maintaining the count of lines on the similar trajactory as one
