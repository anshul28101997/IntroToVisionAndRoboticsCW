#!/usr/bin/env python
import sys
import os
import cv2
import math
import numpy as np
from cv2 import imread
import statistics

im_file = sys.argv[1]
os.system("find_angles.py " + im_file)
from find_angles import yellow_center, red_center, blue_center, green_center, flip
yellow_center = flip(yellow_center)
red_center = flip(red_center)
green_center = flip(green_center)
blue_center = flip(blue_center)

def extractOrangeUpperRegion(image):
	orange_part = cv2.inRange(image, np.array([80,120,140]), np.array([90,140,155]))	
	return orange_part

def extractOrangeLowerRegion(image):
	orange_part = cv2.inRange(image, np.array([20,55,75]), np.array([30,65,85]))	
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
	
def locateBoxTarget(binary_img):
	corner_points = []
	contours = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours[1]:
		if (len(cnt) == 4):
			corner_points = cnt
	top_left = corner_points[0][0]
	top_right = corner_points[1][0]
	bottom_right = corner_points[2][0]
        bottom_left = corner_points[3][0]
	row_cm = (top_left[0] + bottom_left[0]) / 2
	col_cm = (top_left[1] + top_right[1]) / 2
	return (row_cm, col_cm)

def locateSphereTarget(binary_img):
	sphere_centers = []
	sphere_regions = []
	contours = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours[1]:
		# We care about everything except the rectangle
		if (len(cnt) != 4):
			sphere_regions.append(cnt)
	n = len(sphere_regions)
 	for region in sphere_regions:
		sphere_centers.append(np.mean(region, axis=0))
	if (n == 1): # The sphere exists in only one region
		return sphere_centers[0]
	else:
		return (sphere_centers[0]+sphere_centers[1]) / 2

""" CHANGE image2_copy.png to image1_copy and vv if want to see different view of robot """
img = cv2.imread(im_file)

# We need to extract the orange box, not the sphere
# The BGR value of the orange BOX is constant because
# it always stays in the upper region
orange_image_upper = extractOrangeUpperRegion(img)
orange_image_lower = extractOrangeLowerRegion(img)
orange_image = cv2.bitwise_or(orange_image_upper, orange_image_lower)
#cv2.imshow('window33', orange_image)
#cv2.waitKey(10000)
#cm = flip(locateSphereTarget(orange_image))
target = locateSphereTarget(orange_image)[0]
cm_target = flip((int(np.rint(target[0])), int(np.rint(target[1]))))
""" cm IS THE POSITION OF THE CENTER OF MASS OF THE RECTANGLE IN PIXEL VALUES 
    yellow_center IS THE POSITION OF THE CENTER OF MASS OF THE BASE OF THE ROBOT """

def distance(x,y):
	x1,y1 = x
	x2,y2 = y
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def pixel2meter():
	# Take first estimate from link 1
	dist_yellow_blue = distance(yellow_center, blue_center)
	e1 = 2 / dist_yellow_blue
	# Take second estimate from link 3
	dist_blue_green = distance(blue_center, green_center)
	e2 = 3 / dist_blue_green
	# Take third estimate from link 4
	dist_green_red = distance(green_center, red_center)
	e3 = 2 / dist_blue_green
	return (e1+e2+e3)/3

# The code below is just for checking if the images are coming out as I expected
# print to image
img[cm_target] = (255,255,255) 
img[yellow_center] = (255,255,255)
img[blue_center] = (255,255,255)
img[red_center] = (255,255,255)
img[green_center] = (255,255,255)
cv2.imshow('window',img)
cv2.waitKey(20000)
print(cm_target)
print(yellow_center)
#cv2.imshow('window',orange_image)
#cv2.waitKey(5000)

#cv2.imwrite('orange.png',orange_image)
#cv2.waitKey(5000)
