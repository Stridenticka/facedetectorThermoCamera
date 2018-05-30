#import required libraries
#import OpenCV library
import cv2
#import matplotlib library
import inline as inline
import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time
#%matplotlib inline

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

test1 = cv2.imread('data/test1.jpg')
#convert the test image to gray image as opencv face detector expects gray images
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)