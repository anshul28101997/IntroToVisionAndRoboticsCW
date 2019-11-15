#!/usr/bin/env python
import sys
#import os
import cv2
import math
import numpy as np
from cv2 import imread
import statistics
from find_angles import centers, angles, flip

#im_number = 0
#im_file = sys.argv[1]
#if (im_file == 'image1_copy.png'):
#	im_number = 1
#else:
#	im_number = 2
#os.system("find_angles.py " + im_file)
#from find_angles import yellow_center, red_center, blue_center, green_center, flip
#yellow_center = flip(yellow_center)
#red_center = flip(red_center)
#green_center = flip(green_center)
#blue_center = flip(blue_center)

# Save the center locations to a file
#np.save('yc_' + str(im_number), yellow_center)
#np.save('bc_' + str(im_number), blue_center)
#np.save('gc_' + str(im_number), green_center)
#np.save('rc_' + str(im_number), red_center)

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
		if (len(cnt) not in range(0,6)):
			sphere_regions.append(cnt)
	n = len(sphere_regions)
 	for region in sphere_regions:
		sphere_centers.append(np.mean(region, axis=0))
	if (n == 1): # The sphere exists in only one region
		return sphere_centers[0]
	else:
		return (sphere_centers[0]+sphere_centers[1]) / 2

""" CHANGE image2_copy.png to image1_copy and vv if want to see different view of robot """
#img = cv2.imread(im_file)

# We need to extract the orange box, not the sphere
# The BGR value of the orange BOX is constant because
# it always stays in the upper region
#orange_image_upper = extractOrangeUpperRegion(img)
#orange_image_lower = extractOrangeLowerRegion(img)
#orange_image = cv2.bitwise_or(orange_image_upper, orange_image_lower)

#cv2.imshow('window33', orange_image)
#cv2.waitKey(10000)
#cm = flip(locateSphereTarget(orange_image))
#target = locateSphereTarget(orange_image)[0]
#cm_target = flip((int(np.rint(target[0])), int(np.rint(target[1]))))
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

def getCenters(img,img_index):
	orange_image_upper = extractOrangeUpperRegion(img)
	orange_image_lower = extractOrangeLowerRegion(img)
	orange_image = cv2.bitwise_or(orange_image_upper, orange_image_lower)
	target = locateSphereTarget(orange_image)[0]
	cm_target = flip((int(np.rint(target[0])), int(np.rint(target[1]))))
	yellow_center = flip(centers[img_index - 1][0])
	blue_center = flip(centers[img_index - 1][1])
	green_center = flip(centers[img_index - 1][2])
	red_center = flip(centers[img_index - 1][3])
	return [cm_target, yellow_center, blue_center, green_center, red_center]
# The code below is just for checking if the images are coming out as I expected
# print to image
# cv2.imshow('w', orange_image)
#img[cm_target] = (255,255,255) 
#img[yellow_center] = (255,255,255)
#img[blue_center] = (255,255,255)
#img[red_center] = (255,255,255)
#img[green_center] = (255,255,255)
# cv2.imshow('window',img)
# cv2.waitKey(10000)
# print(cm_target)
# print(yellow_center)
#np.save('tc_' + str(im_number),cm_target)
def getCamCenters():
	im1_file = 'image1_copy.png'
	im2_file = 'image2_copy.png'
	img1 = cv2.imread(im1_file)
	img2 = cv2.imread(im2_file)
	r1 = getCenters(img1,1)
	r2 = getCenters(img2,2)
	return [r1, r2]

def get3Dcoordinates(center1, center2):
	z1 = center1[0]
	z2 = center2[0]
	center_cam = np.array([center2[1],center1[1],(z1+z2)/2])
	return center_cam

""" 
param: center_matrix: 
						A 2 by 5 matrix, where the first row is the centers in format (t,y,b,g,r) from camera1 and the second row is the same 							but for camera2.

return:		
						A 1 by 5 matrix, whose elements are the centers in format (t,y,b,g,r) in 3D
"""
def convertCentersTo3D(center_matrix):
	cm_target_3d = get3Dcoordinates(center_matrix[0][0], center_matrix[1][0])
	yellow_center_3d = get3Dcoordinates(center_matrix[0][1], center_matrix[1][1])
	blue_center_3d = get3Dcoordinates(center_matrix[0][2], center_matrix[1][2])
	green_center_3d = get3Dcoordinates(center_matrix[0][3], center_matrix[1][3])	
	red_center_3d = get3Dcoordinates(center_matrix[0][4], center_matrix[1][4])
	return [cm_target_3d, yellow_center_3d, blue_center_3d, green_center_3d, red_center_3d]

def getTargetPosWRTBase():
	centers = convertCentersTo3D(getCamCenters())
	target_pos_wrt_base = (centers[0] - centers[1]) * 0.0345
	target_pos_wrt_base[2] = -target_pos_wrt_base[2] 
	return target_pos_wrt_base
#print(pixel2meter())
#cv2.imshow('window',orange_image)
#cv2.waitKey(5000)quit

#cv2.imwrite('orange.png',orange_image)
#cv2.waitKey(5000)
