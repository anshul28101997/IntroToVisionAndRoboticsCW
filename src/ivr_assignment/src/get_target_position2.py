import numpy as np

#target_center_cam1 = np.load('tc_1.npy')
#target_center_cam2 = np.load('tc_2.npy')
#yellow_center_cam1 = np.load('yc_1.npy')
#yellow_center_cam2 = np.load('yc_2.npy')
#blue_center_cam1 = np.load('bc_1.npy')
#blue_center_cam2 = np.load('bc_2.npy')
#green_center_cam1 = np.load('gc_1.npy')
#green_center_cam2 = np.load('gc_2.npy')
#red_center_cam1 = np.load('rc_1.npy')
#red_center_cam2 = np.load('rc_2.npy')
# estimate the z axis value as the mean
def get3Dcoordinates(center1, center2):
	z1 = center1[0]
	z2 = center2[0]
	center_cam = np.array([center2[1],center1[1],(z1+z2)/2])
	return center_cam
target_center = get3Dcoordinates(target_center_cam1, target_center_cam2)
yellow_center = get3Dcoordinates(yellow_center_cam1, yellow_center_cam2)
blue_center = get3Dcoordinates(blue_center_cam1, blue_center_cam2)
green_center = get3Dcoordinates(green_center_cam1, green_center_cam2)
red_center = get3Dcoordinates(red_center_cam1, red_center_cam2)
print(target_center, yellow_center, blue_center, green_center, red_center)
target_wrt_base = target_center - yellow_center
target_wrt_base[2] = -target_wrt_base[2]

print(target_wrt_base*0.0345)
