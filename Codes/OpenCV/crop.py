# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:27:49 2022

@author: adeel
"""

import cv2


path = 'C:/Users/adeel/Python/rain/cv/'
hd_path_output ="C:/Users/adeel/Python/rain/cv/"


img0 =cv2.imread('C:/Users/adeel/Python/rain/cv/0.jpg')
img1 =cv2.imread('C:/Users/adeel/Python/rain/cv/1.jpg')
img2 =cv2.imread('C:/Users/adeel/Python/rain/cv/2.jpg')
img3 =cv2.imread('C:/Users/adeel/Python/rain/cv/3.jpg')
img4 =cv2.imread('C:/Users/adeel/Python/rain/cv/4.jpg')

input_img = img0


half = int((input_img.shape[1])/2)
left_half, right_half = input_img[:, :half], input_img[:, half:]

 
cv2.imshow('Original_Image', img0)
cv2.imshow('Left_img', left_half)
cv2.imshow('Right_img', right_half)

cv2.waitKey(0)
