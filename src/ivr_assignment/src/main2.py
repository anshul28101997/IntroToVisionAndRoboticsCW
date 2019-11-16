import numpy as np
import cv2
import matplotlib.pyplot as plt
#from get_target_position import convertCentersTo3D
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

target_positions = []
centersIn3D = []
data1, data2 = adjustSize(np.load('c1.npy'), np.load('c2.npy'))
size,_,_ = data1.shape
for i in range(0,size):
	a = convertCentersTo3D([data1[i],data2[i]])
	centersIn3D.append(a)
	target_pos_wrt_base = (a[0] - a[5]) * 0.0345
	target_pos_wrt_base[2] = -target_pos_wrt_base[2] 
	target_positions.append(target_pos_wrt_base)
target_positions = np.asarray(target_positions)

x_values = target_positions[:,0]
y_values = target_positions[:,1]
z_values = target_positions[:,2]
np.save('xs.npy',x_values)
np.save('ys.npy',x_values)
np.save('zs.npy',x_values)
# For now, assume that target_pos_wrt_base is correct
plt.figure(1)
plt.plot(x_values)
plt.plot(y_values)
plt.plot(z_values)
plt.legend(['x','y','z'])
plt.xlabel('Time')
plt.ylabel('Co-ordinates wrt base')

plt.show()
