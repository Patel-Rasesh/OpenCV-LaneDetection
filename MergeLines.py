# TODO
# 1. Number and order of parameters to the functions
# 2. Create more helper functions
# 3. Why vertical and horizontal groups

import math

def distanceFromLine(point, line):
    '''
    Calculating distance of a point from a line
    '''
    x1, y1, x2, y2 = line
    # Using Cartesian coordinates
    x0, y0 = point
    numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + (x2*y1) - (y2*x1))
    #numerator = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
    denominator = pow(y2-y1, 2) + pow(x2-x1, 2)
    return numerator/math.sqrt(denominator)

def computeDistance(line1, line2):
    '''
    Finding the least distance between two lines (between each point of each
    and the other line)
    '''
    return min (distanceFromLine(line1[:2], line2),
                distanceFromLine(line2[:2], line1),
                distanceFromLine(line1[2:], line2),
                distanceFromLine(line2[2:], line1))

def computeOrientation(line):
    '''
    Calculates the slope of a line between pi to -pi
    '''
    x1, y1, x2, y2 = line
    # Calculates angle in Euclidean plane
    linePosition = math.atan2(abs((x1 - x2)), abs((y1 - y2)))
    # Conversion from radians to degree
    return math.degrees(linePosition)
    
def isLineSimilar(newLine, groups, minDistance, minAngle):
    '''
    If line is similar to what we have encountered before, group it with existing lines
    '''
    # 1. Iterate through already existing groups
    for eachGroup in groups:
        # 2. Iterate through each line in the group
        for oldLine in eachGroup:
            # 3a. Check if the distance between lines is less than the threshold
            if computeDistance(newLine, oldLine) < minDistance:
                newLinePosition = computeOrientation(newLine)
                oldLinePosition = computeOrientation(oldLine)
                # 3b. Check if the angle is less than the threshold
                if abs(newLinePosition-oldLinePosition) < minAngle:
                    # 4. Reaching here, the new line qualifies to be in one of the existing groups
                    eachGroup.append(newLine)
                    # 5. Line is not different
                    return False
    # In case, the line is entirely mismatched
    return True

def lineExtremes(rawLines):
    '''
    Fetch the closest and farest points of the group
    '''
    if (len(rawLines) == 1):
        return [rawLines[0][:2], rawLines[0][2:]]
    
    # Grouping points in pair instead of a list of four elements
    pointWise = []
    for line in rawLines:
        # Grouping (x1,y1)
        pointWise.append(line[:2])
        # Grouping (x2,y2)
        pointWise.append(line[2:])

    alignment = computeOrientation(rawLines[0])
    # Checking whether the line falls in vertical of horizontal group of lines
    if 45 < alignment < 135:
        pointWise = sorted(pointWise, key = lambda point:point[1])
    else:
        pointWise = sorted(pointWise, key = lambda point:point[0])

    return [pointWise[0], pointWise[-1]]
def baseLine(rawLines):
    '''
    Forming the initial group and merging them with the help of isLineSimilar()
    '''
    groups = []
    groups.append([rawLines[0]])
    
    # The following parameters we need to tune according to our need
    minDistance = 20
    minAngle = 20

    for nextLine in rawLines[1:]:
        if isLineSimilar(nextLine, groups, minDistance, minAngle):
            groups.append([nextLine])
    
    return groups

def driverMergeLines(rawLines, img):
    '''
    Takes lines from HoughLinesP() and process them to merge them
    '''
    horizontalLines, verticalLines = [], []
    for eachLine in [line[0] for line in rawLines]:
            alignment = computeOrientation(eachLine)
            if 45 < alignment < 135:
                verticalLines.append(eachLine)
            else:
                horizontalLines.append(eachLine)
    verticalLines = sorted(verticalLines, key = lambda line: line[1])
    horizontalLines = sorted(horizontalLines, key = lambda line: line[0])

    mergedLines = []
    for eachLine in [horizontalLines, verticalLines]:
        if len(eachLine) > 0:
            groups = baseLine(eachLine)
            tempMergedLines = []
            for group in groups:
                tempMergedLines.append(lineExtremes(group))
            mergedLines.extend(tempMergedLines)
        
    return mergedLines