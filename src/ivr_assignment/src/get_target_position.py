#!/usr/bin/env python

import cv2
import math
import numpy as np
from cv2 import imread
import statistics
from find_angles import yellow_center

def extractOrange(image):
	orange_part = cv2.inRange(image, np.array([80,120,140]), np.array([90,140,155]))
	return orange_part

def findMean(img):
	# convert the grayscale image to binary image
	ret,thresh = cv2.threshold(img,127,255,0)
	# calculate moments of binary image
	M = cv2.moments(thresh)
 
	# calculate x,y coordinate of center
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	return (cX,cY)
	
def locateTarget(binary_img):
	contours = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	corner_points = contours[1][0]
	top_left = corner_points[0][0]
	top_right = corner_points[1][0]
	bottom_right = corner_points[2][0]
        bottom_left = corner_points[3][0]
	row_cm = (top_left[0] + bottom_left[0]) / 2
	col_cm = (top_left[1] + top_right[1]) / 2
	return (row_cm, col_cm)


""" CHANGE image2_copy.png to image1_copy and vv if want to see different view of robot """
img = cv2.imread("image2_copy.png")

# We need to extract the orange box, not the sphere
# The BGR value of the orange BOX is constant because
# it always stays in the upper region
orange_image = extractOrange(img)
cm = locateTarget(orange_image)
# invert
a,b = cm
cm = (b,a)
c,d = yellow_center
yellow_center = (d,c)

# print to image
img[cm] = (255,255,255)
img[yellow_center] = (255,255,255)

cv2.imshow('window',img)
cv2.waitKey(5000)

#cv2.imshow('window',orange_image)
#cv2.waitKey(5000)

#cv2.imwrite('orange.png',orange_image)
#cv2.waitKey(5000)

""" cm IS THE POSITION OF THE CENTER OF MASS OF THE RECTANGLE IN PIXEL VALUES """


