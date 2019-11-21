import sys
def command(theta, joint):
	return "rostopic pub -1 /robot/joint" + str(joint) + "_position_controller/command std_msgs/Float64 'data: " + str(theta) + "'" 

if __name__ == "__main__":
	theta = float(sys.argv[1])
	joint = int(sys.argv[2])
	cmd = command(theta, joint)
	
