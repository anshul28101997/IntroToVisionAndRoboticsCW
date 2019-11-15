#!/usr/bin/python
import logging
import threading
import time
import numpy as np
import image1
import image2

#global variables
yellow_position = np.array([0,0,0])
blue_position = np.array([0,0,3])
green_position = np.array([0,0,5])
red_position = np.array([0,0,8]) #starting position of the end effector

target_position = np.array([6,2,-5]) #target #orange-red positions #orange position with respect to the red
tetha = np.array([0,0,0,0])



def getLatestOrangeCenter():
    global target_position
    while True:
        center1 = [0,0] # image1.py #YZ axis
        target_position = [target_position[0], center1[0], center1[1]]
        center2 = [0,0] # image2.py #XZ axis
        target_position = [center2[0], target_position[1], center1[1]]
        print(target_position)



if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    threads = list()
    x = threading.Thread(target=image1, args=())
    threads.append(x)
    x.start()
    x = threading.Thread(target=image2, args=())
    threads.append(x)
    x.start()
    x = threading.Thread(target=getLatestOrangeCenter, args=())
    threads.append(x)
    x.start()



