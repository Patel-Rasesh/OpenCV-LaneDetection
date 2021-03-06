import math

def get_orientation(line):
    '''get orientation of a line, using its length
    rxp - computes angle between -pi to pi
    '''
    orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
    return math.degrees(orientation)

def checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
    '''Check if line have enough distance and angle to be count as similar
    '''
    for group in groups:
        # walk through existing line groups
        for line_old in group:
            # check distance
            if get_distance(line_old, line_new) < min_distance_to_merge:
                # check the angle between lines
                orientation_new = get_orientation(line_new)
                orientation_old = get_orientation(line_old)
                # if all is ok -- line is similar to others in group
                if abs(orientation_new - orientation_old) < min_angle_to_merge:
                    group.append(line_new)
                    return False
    # if it is totally different line
    return True

#def DistancePointLine(point, line):
    """Get distance between point and line
    """
    px, py = point
    x1, y1, x2, y2 = line

    def lineMagnitude(x1, y1, x2, y2):
        'Get line (aka vector) length'
        lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        return lineMagnitude

    LineMag = lineMagnitude(x1, y1, x2, y2)
    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine
def distance_to_line(point, line):
    """Get distance between point and line
    https://stackoverflow.com/questions/40970478/python-3-5-2-distance-from-a-point-to-a-line
    """
    px, py = point
    x1, y1, x2, y2 = line
    x_diff = x2 - x1
    y_diff = y2 - y1
    num = abs(y_diff * px - x_diff * py + x2 * y1 - y2 * x1)
    den = math.sqrt(y_diff**2 + x_diff**2)
    return num / den

def get_distance(a_line, b_line):
    """Get all possible distances between each dot of two lines and second line
    return the shortest
    """
    dist1 = distance_to_line(a_line[:2], b_line)
    dist2 = distance_to_line(a_line[2:], b_line)
    dist3 = distance_to_line(b_line[:2], a_line)
    dist4 = distance_to_line(b_line[2:], a_line)

    return min(dist1, dist2, dist3, dist4)
#def get_distance(a_line, b_line):
    """Get all possible distances between each dot of two lines and second line
    return the shortest
    """
    dist1 = DistancePointLine(a_line[:2], b_line)
    dist2 = DistancePointLine(a_line[2:], b_line)
    dist3 = DistancePointLine(b_line[:2], a_line)
    dist4 = DistancePointLine(b_line[2:], a_line)

    return min(dist1, dist2, dist3, dist4)

def merge_lines_pipeline_2(lines):
    'Clusterize (group) lines'
    groups = []  # all lines groups are here
    # Parameters to play with
    min_distance_to_merge = 20
    min_angle_to_merge = 20
    # first line will create new group every time
    groups.append([lines[0]])
    # if line is different from existing gropus, create a new group
    for line_new in lines[1:]:
        if checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
            groups.append([line_new])

    return groups

def merge_lines_segments1(lines):
    """Sort lines cluster and return first and last coordinates
    """
    orientation = get_orientation(lines[0])

    # special case
    if(len(lines) == 1):
        return [lines[0][:2], lines[0][2:]]

    # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
    points = []
    for line in lines:
        points.append(line[:2])
        points.append(line[2:])
    # if vertical
    if 45 < orientation < 135:
        #sort by y
        points = sorted(points, key=lambda point: point[1])
    else:
        #sort by x
        points = sorted(points, key=lambda point: point[0])

    # return first and last point in sorted group
    # [[x,y],[x,y]]
    return [points[0], points[-1]]

def process_lines(lines, img):
    '''Main function for lines from cv.HoughLinesP() output merging
    for OpenCV 3
    lines -- cv.HoughLinesP() output
    img -- binary image
    '''
    lines_x = []
    lines_y = []
    # for every line of cv2.HoughLinesP()
    for line_i in [l[0] for l in lines]:
            orientation = get_orientation(line_i)
            # if vertical
            if 45 < orientation < 135:
                lines_y.append(line_i)
            else:
                lines_x.append(line_i)

    lines_y = sorted(lines_y, key=lambda line: line[1])
    lines_x = sorted(lines_x, key=lambda line: line[0])
    merged_lines_all = []

    # for each cluster in vertical and horizantal lines leave only one line
    for i in [lines_x, lines_y]:
            if len(i) > 0:
                groups = merge_lines_pipeline_2(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(merge_lines_segments1(group))

                merged_lines_all.extend(merged_lines)

    return merged_lines_all