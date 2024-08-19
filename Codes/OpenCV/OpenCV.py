# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:21:07 2022

@author: adeel
"""

import cv2


path = 'C:/Users/adeel/Python/rain/cv/'

img0 =cv2.imread('C:/Users/adeel/Python/rain/cv/0.jpg')
img1 =cv2.imread('C:/Users/adeel/Python/rain/cv/1.jpg')
img2 =cv2.imread('C:/Users/adeel/Python/rain/cv/2.jpg')
img3 =cv2.imread('C:/Users/adeel/Python/rain/cv/3.jpg')
img4 =cv2.imread('C:/Users/adeel/Python/rain/cv/4.jpg')

# get dimensions of image


print('Original Dimensions : ',img0.shape)
width, height = 512,256
dim = (width, height)

  
# resize image
resized = cv2.resize(img0, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)

dimensions = img0.shape
 
# height, width, number of channels in image
height = img0.shape[0]
width = img0.shape[1]
channels = img0.shape[2]
 
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)

#sat_img, map_img = resized[:, :256], resized[:, 256:]
half = int((img0.shape[1])/2)
sat_img, map_img = img0[:, :half], img0[:, half:]

 
cv2.imshow('Original_Image', img0)
cv2.imshow('Sat_img', sat_img)
cv2.imshow('map_img', map_img)
cv2.imshow('Resized_Image', resized)

cv2.waitKey(0)

 