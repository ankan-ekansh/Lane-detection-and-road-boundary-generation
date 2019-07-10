import cv2
import numpy as np

def detectEdge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    #Converting RGB to Grayscale

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    lower_bound_yellow = np.array([10, 0, 100], dtype = "uint8")
    upper_bound_yellow = np.array([110, 255, 255], dtype = "uint8")
    # upper_bound_yellow = np.array([30, 200, 200], dtype = "uint8")
    yellow_mask = cv2.inRange(hls, lower_bound_yellow, upper_bound_yellow)
    # temp = cv2.bitwise_and(image, image, mask = yellow_mask)
    cv2.imshow('Yellow Mask', yellow_mask)
    # lower_bound_white = np.array([0, 200, 0], dtype = "uint8")
    lower_bound_white = np.array([0, 160, 0], dtype = "uint8")
    upper_bound_white = np.array([255, 255, 255], dtype = "uint8")
    white_mask = cv2.inRange(hls, lower_bound_white, upper_bound_white)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    cv2.imshow('White Mask', white_mask)
    yellow_and_white_mask_image = cv2.bitwise_and(image, image, mask = mask)
    g_blur = cv2.GaussianBlur(yellow_and_white_mask_image, (5, 5), 0)      #Applying Gaussian Blur to reduce noise and to smoothen the 
    
    # g_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g_blur, 50, 150)
    cv2.imshow('Gray', gray)
    cv2.imshow('HLS', hls)
    cv2.imshow('Yellow and White mask', mask)
    cv2.imshow('Blur', g_blur)
    return edges

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    # print height
    # print width
    triangle = np.array([
        [( (width * 1)/10, height), ((9 * width)/10, height), ((width)/2, (5 * height)/10)]
    ])
    # print triangle
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    cv2.imshow('Triangle', mask)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img

def create_hough_lines(image):
    lines = cv2.HoughLinesP(image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    # lines = cv2.HoughLinesP(image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 200)
    return lines


def make_points(image, line_params):
    slope, intercept = line_params
    # slope = line_params[0]
    # intercept = line_params[1]
    # y1 = int(image.shape[0] * 1/5)
    y1 = long(image.shape[0])
    y2 = long((y1*65)/100)
    x1 = long((y1 - intercept)/slope)
    x2 = long((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines, prev):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        param = np.polyfit((x1, x2), (y1, y2), 1)   #Fitting a polynomial of degree 1 according to the points, returns slope and y intercept
        
        # print param, 'parameters'

        slope = (param[0])
        intercept = (param[1])
        
        #   Now check if the slope corresponds to the left side or the right side
        #   Since Y increases on moving downwards here, the lines on right side of image have a positive slope and the ones on the left side have a negative slope

        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
        
    # print left, 'left'
    # print right, 'right'
    # if(len(left) == 0):
    #     return np.array([[0,0,0,0], [0,0,0,0]])
    # if(len(right) == 0):
    #     return np.array([[0,0,0,0], [0,0,0,0]])

    left_average = np.average(left, axis = 0)
    right_average = np.average(right, axis = 0)

    # print left_average, 'left avg'
    # print right_average, 'right avg'
    
    # if ((np.isnan(left_average) == False) and ((np.isnan(right_average) == False))):

    if(len(left) == 0):
        # return np.array([[0,0,0,0], [0,0,0,0]])
        # left_line = np.array([0,0,0,0])
        left_line = prev[0]
    else:
        left_line = make_points(image, left_average)
        prev[0] = left_line
    if(len(right) == 0):
        # return np.array([[0,0,0,0], [0,0,0,0]])
        # right_line = np.array([0,0,0,0])
        right_line = prev[1]
    else:
        right_line = make_points(image, right_average)
        prev[1] = right_line
    

    return np.array([left_line, right_line])


def display_lines(image, lines):
    img = np.zeros_like(image)
    if lines is not None:
        for line in lines:  # line is a 2D Array, so we reshape it into a 1D array
            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img, (long(x1), long(y1)), (long(x2), long(y2)), (255, 0, 0), 10)  # 4th Argument is BGR Color, 5th is thickness of line


    # img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # # img = np.zeros_like(image)
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 10)

    return img

def combine(lane_image, line_image):
    final = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    return final

img = cv2.imread('test_image4.jpg')
prev = np.array([[0, 0, 0 , 0], [0, 0, 0, 0]])
im = np.copy(img)
img_with_edge = detectEdge(im)
roi = region_of_interest(img_with_edge)
lines = create_hough_lines(roi)

average_lines = average_slope_intercept(im, lines, prev)
line_img = display_lines(im, average_lines)

# line_img = display_lines(im, lines)

final = combine(im, line_img)

cv2.imshow('Original', img)
cv2.imshow('Edges', img_with_edge)
cv2.imshow('Mask', roi)
cv2.imshow('Line Image', line_img)
cv2.imshow('Final', final)

cv2.waitKey(0)
