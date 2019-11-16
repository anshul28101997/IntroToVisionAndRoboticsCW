#!/usr/bin/env python
# Anshul's git code
#import sys
import cv2
import math
import numpy as np
from cv2 import imread
import statistics
#im_file = sys.argv[1]

def extractRed(image):
   	red_part = cv2.inRange(image, np.array([0,0,110]), np.array([5,5,255]))
    	return red_part
def extractBlue(image):
	red_part = cv2.inRange(image, np.array([110,0,0]), np.array([255,5,5]))
	return red_part
def extractGreen(image):
	red_part = cv2.inRange(image, np.array([0,110,0]), np.array([5,255,5]))
	return red_part
def extractYellow(image):
	red_part = cv2.inRange(image, np.array([0,110,110]), np.array([5,255,255]))
	return red_part

def findMean(img):
	# convert the grayscale image to binary image
	ret,thresh = cv2.threshold(img,127,255,0)
	# calculate moments of binary image
	M = cv2.moments(thresh)
 
	# calculate x,y coordinate of center
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	return (cX,cY)
	
def flip(x):
	a,b = x
	return (b,a)

#img = cv2.imread(im_file)

#red_image = extractRed(img)
#red_center = (findMean (red_image))
#blue_image = extractBlue(img)
#blue_center = findMean(blue_image)
#yellow_image = extractYellow(img)
#yellow_center = findMean(yellow_image)
#green_image = extractGreen(img)
#green_center = findMean(green_image)

def getAngles(cs):
	yellow_center, blue_center, green_center, red_center = cs[0], cs[1], cs[2], cs[3]
	theta1 = math.atan2((blue_center[1]-yellow_center[1]) , (blue_center[0]-yellow_center[0]))
	theta2 = math.atan2((green_center[1]-blue_center[1])  , (green_center[0]-blue_center[0]))
	theta2 -= theta1
	theta3 = math.atan2((red_center[1]-green_center[1])   , (red_center[0]-green_center[0]))
	theta3 -= (theta2+theta1)
	return [theta1, theta2, theta3]

def getMeans(img_file):
	img = cv2.imread(img_file)
	red_image = extractRed(img)
	red_center = (findMean (red_image))
	blue_image = extractBlue(img)
	blue_center = findMean(blue_image)
	yellow_image = extractYellow(img)
	yellow_center = findMean(yellow_image)
	green_image = extractGreen(img)
	green_center = findMean(green_image)
	return [yellow_center, blue_center, green_center, red_center]

def runImages():
	c1 = getMeans('image1_copy.png')
	c2 = getMeans('image2_copy.png')
	a1 = getAngles(c1)
	a2 = getAngles(c2)
	return [[c1, c2],[a1,a2]]		

def runImage(img_file):
	c1 = getMeans(img_file)
	a1 = getAngles(c1)
	return [[c1],[a1]]

centers_and_angles = runImages()

centers = centers_and_angles[0]
angles = centers_and_angles[1]
#print(angles)
