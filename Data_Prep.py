import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import cv2
import os
import sys
from cv2 import VideoCapture
from tensorflow.keras.metrics import RootMeanSquaredError

#
# vidcap = VideoCapture("train.mp4")
# os.chdir("Images") # Make this directory first
# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = vidcap.read()
#     if hasFrames:
#         cv2.imwrite("image"+str(count)+".jpeg", image)     # save frame as JPEG file
#     return hasFrames
# sec = 0
# frameRate = 0.05 #//it will capture image in each 0.05 second
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)
