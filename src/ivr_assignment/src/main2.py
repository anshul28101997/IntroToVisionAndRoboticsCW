import numpy as np
import cv2
import matplotlib.pyplot as plt
import rosbag
from get_target_position import distance
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

def get3Dcoordinates(center1, center2):
	z1 = center1[0]
	z2 = center2[0]
	center_cam = np.array([center2[1],center1[1],(z1+z2)/2])
	return center_cam

""" 
		params: Two matrices, each of shape 5 by 2, where the rows represent centers in form (t,y,b,g,r)
		returns: A 5 by 3 matrix, which has the centers in form (t,y,b,g,r) in 3D
"""
#def convertCentersTo3D(mat1, mat2):
#	output = []
#	for i in range(0,5):
#		center = get3Dcoordinates(mat1[i], mat2[i])
#		output.append(center)
#	return output


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

def estimatePixel2Meter(yellow_center, blue_center, green_center, red_center):
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

target_positions = []
centersIn3D = []
data1, data2 = adjustSize(np.load('c1.npy'), np.load('c2.npy'))
size,_,_ = data1.shape

for i in range(0,size):
	a = convertCentersTo3D([data1[i],data2[i]])
	centersIn3D.append(a)
	print(a[0], a[5])
	target_pos_wrt_base = (a[0] - a[5]) * 0.0345
	target_pos_wrt_base[2] = -target_pos_wrt_base[2]
	target_positions.append(target_pos_wrt_base)
target_positions = np.asarray(target_positions)
x_values = target_positions[:,0]
#print(np.asarray(x_values))
y_values = target_positions[:,1]
z_values = target_positions[:,2]
required_size = x_values.shape[0]
#np.save('xs.npy',x_values)
#np.save('ys.npy',y_values)
#np.save('zs.npy',z_values)
# For now, assume that target_pos_wrt_base is correct
#plt.figure(1)
#plt.plot(x_values)
#plt.plot(y_values)
#plt.plot(z_values)
#plt.legend(['x','y','z'])
#plt.xlabel('Time')
#plt.ylabel('Co-ordinates wrt base')
#plt.show()

# How to read a rosbag file
""" Input: file_name of rosbag file
	Ouput: numpy array of numbers (co-ordinates of msg of rosbag)
"""
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
