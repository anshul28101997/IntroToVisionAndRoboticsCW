import numpy as np
import math

#rotation matrix
def Rx(tetha): #tehta needs to be in radians
    s = math.sin (tetha)
    c = math.cos (tetha)
    return np.array([[1,0,0] , [0,c,-s] , [0,s,c]])

def Ry(tetha): #tehta needs to be in radians
    s = math.sin (tetha)
    c = math.cos (tetha)
    return np.array([[c,0,-s] , [0,1,0] , [s,0,c]])

def Rz(tetha): #tehta needs to be in radians
    s = math.sin (tetha)
    c = math.cos (tetha)
    return np.array([[c,-s,0] , [s,c,0] , [0,0,1]])

#find the point you want to move
#standardise it with respect to the point which is going to move (pt.centre)
#move the point along the transformation
#add the center back to the point to get the overall position
def move (center_fixed, center_tomove, tetha, plane):
    center_tomove = center_tomove - center_fixed
    if (plane=='x'):
        center_tomove = Rx(tetha).dot(center_tomove)
    elif (plane=='y'):
        center_tomove = Ry(tetha).dot(center_tomove)
    else:
        center_tomove = Rz(tetha).dot(center_tomove)
    center_tomove = center_tomove + center_fixed
    return center_tomove

def printStatus():
    global yellow_position, blue_position, green_position, red_position
    print('yellow', yellow_position)
    print('blue  ', blue_position)
    print('green ', green_position)
    print('red   ', red_position)
    print('orange', target_position)

#[t1,t1,t3,t4] #where args is a 1x4 vector of the tetha changes. 
#ONLY 1 element must contain value becuase the order of movement matters.
def collaborated_movement (args, yellow_position, blue_position, green_position, red_position): 
    blue_position = move (yellow_position, blue_position, args[0], 'z')
    green_position = move (yellow_position, green_position, args[0], 'z')
    red_position = move (yellow_position, red_position, args[0], 'z')
    
    green_position = move (blue_position, green_position, args[1], 'x')
    red_position = move (blue_position, red_position, args[1], 'x')
    
    green_position = move (blue_position, green_position, args[2], 'y')
    red_position = move (blue_position, red_position, args[2], 'y')
    
    red_position = move (green_position, red_position, args[3], 'x')
    return yellow_position , blue_position, green_position , red_position
