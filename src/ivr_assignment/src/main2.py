import numpy as np
from get_target_position import convertCentersTo3D
def adjustSize(c1,c2):
	size1,_,_ = c1.shape
	size2,_,_ = c2.shape
	if (size1 == size2):
		return (c1,c2)
	else:
		if (size1 > size2):
			# Remove last element of c1
			c1 = c1[list(range(0,size1-1)),:,:]
			adjustSize(c1,c2)
		else:
			# Remove last element of c2
			c2 = c2[list(range(0,size2-1)),:,:]
			adjustSize(c1,c2)

data1 = np.load('c1.npy')
data2 = np.load('c2.npy')
# Hardcoding for now
data2 = data2[range(0,22),:,:]
# *******************
print(data1[0],'**')
print(data2[0],'*')
a = convertCentersTo3D([data1[0],data2[0]])
target_pos_wrt_base = (a[0] - a[1]) * 0.0345
target_pos_wrt_base[2] = -target_pos_wrt_base[2] 
print(target_pos_wrt_base)




