#!/usr/bin/env python
# coding: utf-8

'''

Purpose:

This program is used to analyse a series of stereo images taken 
from two cameras to detect non-linear motion in 3D space. 
Additional libraries used for this are NumPy and OpenCV. 

The program starts by reading the paired images. For every image, 
the program identifies the colored objects and their pixel coordinates 
in the image. Using a pair of images of the same frame, the program 
then calculates the real X, Y and Z coordinates for every object 
in each frame.

Using the coordiantes, the program then constructs a line for 
each object that passes through the start and endpoint of the object. 
It then iterates over each coordinate of the object and compares 
the distance between the coordinate and the line. If the distance 
surpasses the threshold, the object is considered to be a UFO.

In the end, all the Z distances for each object and each 
frame are printed out as well as the UFOs.

The programm starts in the main() method.

'''

# Import additional libraries
import cv2
import sys
import numpy as np
from decimal import Decimal

# Constants

PIXEL_SPACING = 0.00001
# Used to specify the focal length
FOCAL_LENGTH = 12
# Used to specify the distance between two cameras
BASELINE = 3500
# Used to specify the max error allowed for an object to be classes as an asteroid
THRESHOLD_DISTANCE = 200

'''
Constants for color ranges
cyan_range: from COLOR_RANGES[0][0] to COLOR_RANGES[0][1]
white_range: from COLOR_RANGES[1][0] to COLOR_RANGES[1][1]
red_lower_range: from COLOR_RANGES[2][0] to COLOR_RANGES[2][1]
red_upper_range: from COLOR_RANGES[2][2] to COLOR_RANGES[2][3]
yellow_range: from COLOR_RANGES[3][0] to COLOR_RANGES[3][1]
blue_range: from COLOR_RANGES[4][0] to COLOR_RANGES[4][1]
orange_range: from COLOR_RANGES[5][0] to COLOR_RANGES[5][1]
green_range: from COLOR_RANGES[6][0] to COLOR_RANGES[6][1]
'''

COLOR_RANGES = [
    [np.array([85, 40, 40]), np.array([95, 255, 255])],
    [np.array([0, 0, 170]), np.array([255, 12, 255])],
    [np.array([0, 40, 40]), np.array([10, 255, 255]), np.array(
        [170, 40, 40]), np.array([180, 255, 255])],
    [np.array([25, 40, 40]), np.array([43, 255, 255])],
    [np.array([100, 40, 40]), np.array([120, 255, 255])],
    [np.array([10, 40, 40]), np.array([25, 255, 255])],
    [np.array([40, 40, 40]), np.array([70, 255, 255])]
]

'''
A function to find colored objects and their coordinates on the image.
:param array img: an image read using cv2.imread()
:return: a dictionary containing each identity and its corresponding coordinates
'''


def findCoordinates(img):
    # Convert image from BGR to HSV.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    objects = {}
    # Iterate over all color ranges to find it in the image.
    # The method used for detecting colored objects is adapted from
    # 	the method explained in the link below.
    # https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
    for count, color_range in enumerate(COLOR_RANGES):
        # Special case for red as it has two ranges.
        if (count == 2):
            mask1 = cv2.inRange(hsv, color_range[0], color_range[1])
            mask2 = cv2.inRange(hsv, color_range[2], color_range[3])
            mask = mask1 + mask2
        else:
            mask = cv2.inRange(hsv, color_range[0], color_range[1])

        # Find contours based on the mask.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # If there are no contours, skip the iteration.
        if (len(contours) == 0):
            continue

        # Select the contour with the highest number of contours.
        # This is done so that a white reflection that appears in
            # 	the last images is not classed as white.
        contour = max(contours, key=len)

        # Finding the coordinates of an object.
        # Using the x and y of the contour if it only returns 1.
        if(len(contour) == 1):
            x = int(contour[0][0][0])
            y = int(contour[0][0][1])
        # Finding the mid-distance if there are only 2 contours.
        elif(len(contour) == 2):
            x = int((contour[0][0][0] + contour[1][0][0]) / 2)
            y = int((contour[0][0][1] + contour[1][0][1]) / 2)
        else:
            # Using OpenCV moments for multiple contours.
            # The formula used for this is explained and taken from the link below.
            # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
            moment = cv2.moments(contour)
            if(moment['m00'] != 0):
                x = int(moment['m10'] / moment['m00'])
                y = int(moment['m01'] / moment['m00'])

        # Assigning the correct distance to a corresponding object.
        if (count == 0):
            objects['cyan'] = [x, y]
        elif (count == 1):
            objects['white'] = [x, y]
        elif (count == 2):
            objects['red'] = [x, y]
        elif (count == 3):
            objects['yellow'] = [x, y]
        elif (count == 4):
            objects['blue'] = [x, y]
        elif (count == 5):
            objects['orange'] = [x, y]
        elif (count == 6):
            objects['green'] = [x, y]

    return objects


'''
A function to find real coordinates of all objects.
:param string left_img_ref: a string path to the left image
:param string right_img_ref: a string path to the right image
:return: a dictionary containing each identity and its corresponding coordinates
'''


def findRealCoordinates(left_img_ref, right_img_ref):
    # Read the left and right images using the references.
    left_img = cv2.imread('./images/' + left_img_ref)
    right_img = cv2.imread('./images/' + right_img_ref)

    # Find objects and their coordinates in both images.
    left_objects = findCoordinates(left_img)
    right_objects = findCoordinates(right_img)

    objects = {}

    # Select the dictionary with the least objects detected.
    # This is done so that if one image is missing an object,
    #	 the object gets ignored
    min_object_dic = min(left_objects, right_objects, key=len)

    # Iterate over the dictionary with the least objects detected.
    for key, value in min_object_dic.items():
        # Check if an object is detected in both images.
        if key in left_objects and key in right_objects:

            # Compute the real X, Y values.
            realLeftX, realLeftY = convertPixelsToCoordinates(left_img,
                                                              left_objects[key][0], left_objects[key][1])
            realRightX, realRightY = convertPixelsToCoordinates(right_img,
                                                                right_objects[key][0], right_objects[key][1])

            # Compute the Z distance using the formula described in
            # 	unit 9 of lecture slides.
            x_diff = abs(realLeftX - realRightX)
            Z = int((FOCAL_LENGTH * BASELINE) / (x_diff * PIXEL_SPACING))
            x = int(((realRightX * PIXEL_SPACING) / FOCAL_LENGTH) * Z)
            y = int(((realRightY * PIXEL_SPACING) / FOCAL_LENGTH) * Z)
            objects[key] = [x, y, Z]

    return objects


# Convert the pixel coordinates to real coordinates.
# The formula used for this is described in unit 9 of lecture slides.
def convertPixelsToCoordinates(img, x, y):
    imY, imX, _ = img.shape
    halfX = imX / 2
    halfY = imY / 2
    rX = x - halfX
    rY = halfY - y
    return (rX, rY)


'''
A function to calculate the perpendicular distance between a straight line 
	and a point.
:param array p0: coordinates of a starting point of the line
:param array p1: coordinates of an endpoint of the line
:param array p2: coordinates of a point used to calculate the distance 
	to a line
:return: float that representing the perpendicular distance from a point 
	to a line
'''


def distanceBetweenPointAndLine(p0, p1, p2):
    # The method and the formula are explained and taken from the link below.
    # https://onlinemschool.com/math/library/analytic_geometry/p_line/

    # Firstly, I divide all coordinates by 100 as not doing so results in
    # 	overflow errors.
    p0 = [element / 100 for element in p0]
    p1 = [element / 100 for element in p1]
    p2 = [element / 100 for element in p2]

    # Compute the vector if a line using the NumPy subtract function.
    vector = np.subtract((p0[0], p0[1], p0[2]), (p1[0], p1[1], p1[2]))

    # Compute the M0M1.
    m0m1 = np.subtract((p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2]))

    # Compute the crossproduct using the NumPy cross function.
    m0m1xs = np.cross(m0m1, vector)

    # Compute the distance using the formula.
    distance = (np.sqrt((m0m1xs[0]**2) + (m0m1xs[1]**2) + (m0m1xs[2]**2))) / (
        np.sqrt((vector[0]**2) + (vector[1]**2) + (vector[2]**2)))

    # Multiply the distance by 100 to account for the division at
    #	 the start of the function.
    return distance * 100


'''
A function to find the UFOs.
:param array coordinates: contains all real coordinates of 
	all identified objects in all frames
:return: an array containing all identified UFOs

Firstly, this function finds the starting and ending points for all objects
using the coordinates provided. The function then iterates over all 
coordinates of a particular object and computes the distance from the coordinate 
to the line that passed through the starting and ending point using 
the distanceBetweenPointAndLine function. If the distance surpasses the threshold, 
that means that this particular object is a UFO.
'''


def findUFO(coordinates):
    # Define objects
    keys = ['orange', 'red', 'cyan', 'yellow', 'white', 'blue', 'green']
    starting_coordinates = {}
    ending_coordinates = {}
    ufos = []

    # Find the starting coordinates of all objects.
    # A loop is used as not all objects are in the first frame.
    for coordinate in coordinates:
        # Check if all coordinates are found, break if so.
        if len(starting_coordinates) == len(keys):
            break
        for key in keys:
            # If an object is in the frame and has not been recorded yet.
            if key in coordinate and key not in starting_coordinates:
                starting_coordinates[key] = coordinate[key]

    # Find the ending coordinates of all objects.
    # A loop is used as not all objects are in the last frame.
    for coordinate in reversed(coordinates):
        # Check if all coordinates are found, break if so.
        if len(ending_coordinates) == len(keys):
            break
        for key in keys:
            # If an object is in the frame and has not been recorded yet.
            if key in coordinate and key not in ending_coordinates:
                ending_coordinates[key] = coordinate[key]

    # Loop through all objects to determine which ones are UFOs.
    for key in keys:
        # Define the starting and ending point that are passed to
        # 	distanceBetweenPointAndLine function.
        p0 = [starting_coordinates[key][0], starting_coordinates[key]
              [1], starting_coordinates[key][2]]
        p1 = [ending_coordinates[key][0], ending_coordinates[key]
              [1], ending_coordinates[key][2]]
        # Loop through every coordinate of an object and compute the distance
        # 	from the line to the point.
        for coordinate in coordinates:
            # Check if the object is in the frame, if not skip this frame.
            if key not in coordinate:
                continue
            p2 = [coordinate[key][0], coordinate[key][1], coordinate[key][2]]
            # Calculate the distance
            distance_between_point = distanceBetweenPointAndLine(p0, p1, p2)
            # If the distance surpasses the threshold,
            # 	append to UFOs array and break the loop.
            if (distance_between_point > THRESHOLD_DISTANCE):
                ufos.append(key)
                break
    return ufos


def main():
    # Define the array that will hold all coordinates of
    # 	all identified objects for all frames.
    coordiantes_for_all_frames = []
    nframes = int(sys.argv[1])
    print('frame' + " " + "identity" + " " + "distance")
    print("")
    # Read each frame, identify the objects as well as their coordinates
    # 	and append to an array.
    for frame in range(0, nframes):
        fn_left = sys.argv[2] % frame
        fn_right = sys.argv[3] % frame
        coordiantes = findRealCoordinates(fn_left, fn_right)
        for identity, coordinates in coordiantes.items():
            coordiantes_for_all_frames.append(coordiantes)
            # Print the frame, identify and Z distance.
            print('%5s  %6s  %6s' % (str(frame + 1), identity,
                                     '%.2E' % Decimal(coordinates[2])))

    # Find the UFOs and print them out.
    ufos = findUFO(coordiantes_for_all_frames)
    print("")
    print("UFO: ", *ufos)


main()
