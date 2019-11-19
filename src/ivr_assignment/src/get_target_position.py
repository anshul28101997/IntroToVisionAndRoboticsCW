#!/usr/bin/env python
import sys
#import os
import cv2
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
import rosbag
""" ******************************** THIS PART OF THE CODE IS FOR GETTING THE ANGLES AND THE CENTERS ***************************** """
def flip(x):
	a,b = x
	return (b,a)

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

def extractBaseFrame(image):
	base_part = cv2.inRange(image, np.array([118,118,118]), np.array([123,123,123]))
	return base_part

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
	base_frame = extractBaseFrame(img)
	base_center = findMean(base_frame)
	return [yellow_center, blue_center, green_center, red_center, base_center]

def runImages():
	c1 = getMeans('image1_copy.png')
	c2 = getMeans('image2_copy.png')
	a1 = getAngles(c1)
	a2 = getAngles(c2)
	return [[c1, c2],[a1,a2]]		

""" Get the centers and the angles for ONLY THE ROBOT """
centers_and_angles = runImages()
centers = centers_and_angles[0]
angles = centers_and_angles[1]

""" Now, get the center of the target sphere and all the robot centers, completely processed """

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
	base_center = flip(centers[img_index - 1][4])
	return [cm_target, yellow_center, blue_center, green_center, red_center, base_center]


""" *************************************************************************************************************** """

# FOR NOW ASSUME THE DISTANCE BETWEEN TWO ADJACENT PIXELS IS 0.0345m

""" *********************************************** CONVERSION FROM 2D TO 3D ******************************************** """

def distance(x,y):
	x1,y1 = x
	x2,y2 = y
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)

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
	base_center_3d = get3Dcoordinates(center_matrix[0][5], center_matrix[1][5])
	return [cm_target_3d, yellow_center_3d, blue_center_3d, green_center_3d, red_center_3d, base_center_3d]

def adjustSize(c1,c2):
	size1,_,_ = c1.shape
	size2,_,_ = c2.shape
	if (size1 == size2):
		return (c1,c2)
	else:
		if (size1 > size2):
			# Remove last element of c1
			c1 = c1[list(range(0,size1-1)),:,:]
			return adjustSize(c1,c2)
		else:
			# Remove last element of c2
			c2 = c2[list(range(0,size2-1)),:,:]
			return adjustSize(c1,c2)

""" MAIN """

def parseRosBag(bag_file):
	bag = rosbag.Bag(bag_file)
	co_ordinates = []
	for topic, msg, t in bag.read_messages():
		string = str(msg)
		n = len(string)
		number = float(string[6:n-1])
		co_ordinates.append(number)
	bag.close()
	return co_ordinates

def run():
	target_positions = []
	centersIn3D = []
	data1, data2 = adjustSize(np.load('c1.npy'), np.load('c2.npy'))
	size,_,_ = data1.shape
	for i in range(0,size):
		a = convertCentersTo3D([data1[i],data2[i]])
		centersIn3D.append(a)
		target_pos_wrt_base = (a[0] - a[5]) * 0.0345
		target_pos_wrt_base[2] = -target_pos_wrt_base[2] + 0.5
		target_positions.append(target_pos_wrt_base)
	target_positions = np.asarray(target_positions)
	x_values = target_positions[:,0]
	y_values = target_positions[:,1]
	z_values = target_positions[:,2]
	required_size = x_values.shape[0]

	x_values_true = parseRosBag('x_values.bag')
	x_values_true = x_values_true[0:required_size-1]
	y_values_true = parseRosBag('y_values.bag')
	y_values_true = y_values_true[0:required_size-1]
	z_values_true = parseRosBag('z_values.bag')
	z_values_true = z_values_true[0:required_size-1]

	# Let's plot!

	plt.figure(1)
	plt.plot(x_values_true)
	plt.plot(x_values)
	plt.legend(['True positions','Predicted positions'])
	plt.xlabel('Time')
	plt.ylabel('x co-ordinate')
	plt.show()

	plt.figure(2)
	plt.plot(y_values_true)
	plt.plot(y_values)
	plt.legend(['True positions','Predicted positions'])
	plt.xlabel('Time')
	plt.ylabel('y co-ordinate')
	plt.show()

	plt.figure(3)
	plt.plot(z_values_true)
	plt.plot(z_values)
	plt.legend(['True positions','Predicted positions'])
	plt.xlabel('Time')
	plt.ylabel('z co-ordinate')
	plt.show()

run()
