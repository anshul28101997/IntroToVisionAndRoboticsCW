#!/usr/bin/env python
# Anshul's git code
import cv2
import math
import numpy as np
from cv2 import imread
import statistics


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

def extractOrange(image):
	red_part = cv2.inRange(image, np.array([10,50,70]), np.array([120,140,160]))
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
	

img = cv2.imread("image1_copy.png")

red_image = extractRed(img)
red_center = (findMean (red_image))
blue_image = extractBlue(img)
blue_center = findMean(blue_image)
yellow_image = extractYellow(img)
yellow_center = findMean(yellow_image)
green_image = extractGreen(img)
green_center = findMean(green_image)

# For now, extraction of orange is not working properly
orange_image = extractOrange(img)
cv2.imshow('window',orange_image)
cv2.imwrite('orange.png',orange_image)
cv2.waitKey(2000)

#print(red_center[0])

tetha1 = math.atan2((blue_center[1]-yellow_center[1]) , (blue_center[0]-yellow_center[0]))
tetha2 = math.atan2((green_center[1]-blue_center[1])  , (green_center[0]-blue_center[0]))
tetha2 -= tetha1
#tetha2 %= math.pi
tetha3 = math.atan2((red_center[1]-green_center[1])   , (red_center[0]-green_center[0]))
tetha3 -= (tetha2+tetha1)
#tetha3 %= math.pi

print("tetha1:", tetha1)
print("tetha2:", tetha2)
print("tetha3:", tetha3)
print("errors in angles becuase some portions of the joints were hidden")

