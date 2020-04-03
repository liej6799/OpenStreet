# Library
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import linear_model

#General Settings
image = '0000000001.png'
folder = '../video-data/kitti/2011_09_26_drive_0001_extract/image_00/data/'

def gaussianBlur(image):
    return cv2.GaussianBlur(image,(5,5),0)

def regionOfInterest(image):
    height = image.shape[0]
    triangle = np.array([[(460, height), (1000, height), (650, 450)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(image, mask)
    #return mask


def extract_lane(road_lines):
    left_lane = []
    right_lane = []
    left_slope = []
    right_slope = []

    if road_lines is not None:
        for x in range(0, len(road_lines)):
            for x1, y1, x2, y2 in road_lines[x]:
                slope = compute_slope(x1, y1, x2, y2)
                if slope is None:
                    slope = 0.0
                if (slope < 0):
                    left_lane.append(road_lines[x])
                    left_slope.append(slope)
                else:
                    if (slope > 0):
                        right_lane.append(road_lines[x])
                        right_slope.append(slope)

    return left_lane, right_lane, left_slope, right_slope


def compute_slope(x1, y1, x2, y2):
    if x2 != x1:
        return ((y2 - y1) / (x2 - x1))


def split_append(left_lane, right_lane):
    left_lane_sa = []
    right_lane_sa = []

    for x in range(0, len(left_lane)):
        for x1, y1, x2, y2 in left_lane[x]:
            left_lane_sa.append([x1, y1])
            left_lane_sa.append([x2, y2])

    for y in range(0, len(right_lane)):
        for x1, y1, x2, y2 in right_lane[y]:
            right_lane_sa.append([x1, y1])
            right_lane_sa.append([x2, y2])

    left_lane_sa = np.array(left_lane_sa)
    right_lane_sa = np.array(right_lane_sa)
    left_lane_sa, right_lane_sa = sort(left_lane_sa, right_lane_sa)
    return left_lane_sa, right_lane_sa


def sort(left_lane_sa, right_lane_sa):
    left_lane_sa = left_lane_sa[np.argsort(left_lane_sa[:, 0])]
    right_lane_sa = right_lane_sa[np.argsort(right_lane_sa[:, 0])]

    return left_lane_sa, right_lane_sa


def ransac_drawlane(left_lane_sa, right_lane_sa, frame):
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []

    for x1, y1 in left_lane_sa:
        left_lane_x.append([x1])
        left_lane_y.append([y1])

    for x1, y1 in right_lane_sa:
        right_lane_x.append([x1])
        right_lane_y.append([y1])

    left_ransac_x = np.array(left_lane_x)
    left_ransac_y = np.array(left_lane_y)

    right_ransac_x = np.array(right_lane_x)
    right_ransac_y = np.array(right_lane_y)

    left_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    # print(left_ransac_x,left_ransac_y,len(left_ransac_x),len(left_ransac_y), left_ransac_x.shape )
    left_ransac.fit(left_ransac_x, left_ransac_y)
    slope_left = left_ransac.estimator_.coef_
    intercept_left = left_ransac.estimator_.intercept_

    right_ransac = linear_model.RANSACRegressor()
    right_ransac.fit(right_ransac_x, right_ransac_y)
    slope_right = right_ransac.estimator_.coef_
    intercept_right = right_ransac.estimator_.intercept_

    ysize = frame.shape[0]
    xsize = frame.shape[1]
    y_limit_low = int(0.95 * ysize)
    y_limit_high = int(0.65 * ysize)

    # Coordinates for point 1(Bottom Left)
    y_1 = ysize
    x_1 = int((y_1 - intercept_left) / slope_left)

    # Coordinates for point 2(Bottom Left)
    y_2 = y_limit_high
    x_2 = int((y_2 - intercept_left) / slope_left)

    # Coordinates for point 3(Bottom Left)
    y_3 = y_limit_high
    if slope_right == 0:
        slope_right = 1
    x_3 = int((y_3 - intercept_right) / slope_right)

    # Coordinates for point 4(Bottom Right)
    y_4 = ysize
    x_4 = int((y_4 - intercept_right) / slope_right)

    cv2.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 255), 3)
    cv2.line(frame, (x_3, y_3), (x_4, y_4), (0, 255, 255), 3)
    pts = np.array([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]])
    mask_color = (255, 255, 0)
    frame_copy = frame.copy()
    cv2.fillPoly(frame_copy, np.int32([pts]), mask_color)
    opacity = 0.4
    cv2.addWeighted(frame_copy, opacity, frame, 1 - opacity, 0, frame)
    return frame



def getGrayImage():
    # Convert to Grayscale
    return cv2.cvtColor(getImage(), cv2.COLOR_BGR2GRAY)

cap  = cv2.VideoCapture('/home/joes/Downloads/video.mp4')
while True:
    _, frame = cap.read()
    #imgHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # Color space conversion
    roi_image = (regionOfInterest(frame))
    img_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    mask_white = cv2.inRange(img_gray, 110, 255)


    canny_image = cv2.Canny(roi_image, 100, 200)
    mask_onimage = cv2.bitwise_and(img_gray, canny_image)


    lines = cv2.HoughLinesP(mask_onimage, 1, np.pi/180, 50, maxLineGap=50)
    left_lane, right_lane, left_slope, right_slope = extract_lane(lines)
    left_lane_sa, right_lane_sa = split_append(left_lane, right_lane)
    image_np = ransac_drawlane(left_lane_sa, right_lane_sa,frame)

    cv2.imshow("awd", image_np)

    #cv2.imshow("awd", image_np)

    key = cv2.waitKey(1)
    if key == 27:
        break


#

#
#




#cv2.imshow("awd", image_np)
#cv2.waitKey(0)

