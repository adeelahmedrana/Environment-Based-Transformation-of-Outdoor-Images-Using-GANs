# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:12:07 2022

@author: adeel
"""

import glob
import os
import cv2
com_path ="/home/ahd7rng/DS/Dataset/c/clean commuter data/"
lidar_path ="/home/ahd7rng/DS/Dataset/lidar_grid/"
sd_path = "/home/ahd7rng/DS/Dataset/sd_map/"
hd_path = "/home/ahd7rng/DS/Dataset/hd_map/"
output_path = "/home/ahd7rng/DS/Dataset/sd_map and Velabs cleanned/"
output_hd_path = "/home/ahd7rng/DS/Dataset/output_hd_path/"
com_names = [x for x in glob.glob(com_path+"*.png")]
lidar_names = [x for x in glob.glob(lidar_path+"*.png")]
sd_names = [x for x in glob.glob(sd_path+ "*.png")]
hd_names = [x for x in glob.glob(hd_path+ "*.png")]
com_names
print(os.path.basename(sd_names[0]))
132616.png
len(sd_names)
16081
len(lidar_names)
6662
len(com_names)
589
for c in com_names:
    image_name = os.path.basename(c)
    sd_img = sd_path +image_name
    ld_img = lidar_path +image_name
    com_image= cv2.imread(c)
    sd_image= cv2.imread(sd_img)
    lidar_image =cv2.imread(ld_img)
    
    #spliting SD Map Images into BGR
    blue,green,red = cv2.split(sd_image)
        
    #concat_image = cv2.hconcat([sd_image,hd_image])
    #gray1 = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(com_image, cv2.COLOR_BGR2GRAY)


    img_merged = cv2.merge((green,red,gray3))
    cv2.imwrite(output_path +str(os.path.basename(c)), img_merged)
for c in com_names:
    image_name = os.path.basename(c)
    #sd_img = sd_path +image_name
    hd_img = hd_path +image_name

    #ld_img = lidar_path +image_name
    com_image= cv2.imread(c)
    hd_image= cv2.imread(hd_img)
    #lidar_image =cv2.imread(ld_img)
        
    #concat_image = cv2.hconcat([sd_image,hd_image])
    #gray1 = cv2.cvtColor(com_image, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(sd_image, cv2.COLOR_BGR2GRAY)
    #gray3 = cv2.cvtColor(lidar_image, cv2.COLOR_BGR2GRAY)


   # img_merged = cv2.merge((gray1,gray2,gray3))
    cv2.imwrite(output_hd_path +str(os.path.basename(c)), hd_image)

*************************************


Concat Images
import glob
import os
import cv2
hd_path_output ="/home/ahd7rng/DS/Dataset/hd dilate 3/*.png"
output ="/home/ahd7rng/DS/Dataset/output/*.png"
output_concat_path = "/home/ahd7rng/DS/Dataset/Output dil 3/"
hd_names = [x for x in glob.glob(hd_path_output)]
sd_names = [x for x in glob.glob(output)]
hd_names
print(os.path.basename(hd_names[0]))
130798.png
len(sd_names)
589
for h,s in zip(hd_names,sd_names):
    if os.path.basename(h) == os.path.basename(s):
        print("same name")
        
        sd_image= cv2.imread(s)
        hd_image =cv2.imread(h)
        concat_image = cv2.hconcat([hd_image,sd_image])
        cv2.imwrite(output_concat_path +str(os.path.basename(h)), concat_image)
        
    else:
        print("Filename not same")
        print(h,s)
        

Channel merge 4 3 images
import glob
import os
import cv2
com_path ="/home/ahd7rng/DS/Dataset/c/clean commuter data/"
lidar_path ="/home/ahd7rng/DS/Dataset/lidar_grid/"
sd_path = "/home/ahd7rng/DS/Dataset/sd_map/"
hd_path = "/home/ahd7rng/DS/Dataset/hd_map/"
output_path = "/home/ahd7rng/DS/Dataset/sd_map and Velabs cleanned/"
output_hd_path = "/home/ahd7rng/DS/Dataset/output_hd_path/"
com_names = [x for x in glob.glob(com_path+"*.png")]
lidar_names = [x for x in glob.glob(lidar_path+"*.png")]
sd_names = [x for x in glob.glob(sd_path+ "*.png")]
hd_names = [x for x in glob.glob(hd_path+ "*.png")]
com_names
for c in com_names:
    image_name = os.path.basename(c)
    sd_img = sd_path +image_name
    ld_img = lidar_path +image_name
    com_image= cv2.imread(c)
    sd_image= cv2.imread(sd_img)
    lidar_image =cv2.imread(ld_img)
    
    #spliting SD Map Images into BGR
    blue,green,red = cv2.split(sd_image)
        
    #concat_image = cv2.hconcat([sd_image,hd_image])
    #gray1 = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(com_image, cv2.COLOR_BGR2GRAY)


    img_merged = cv2.merge((green,red,gray3))
    cv2.imwrite(output_path +str(os.path.basename(c)), img_merged)
for c in com_names:
    image_name = os.path.basename(c)
    #sd_img = sd_path +image_name
    hd_img = hd_path +image_name

    #ld_img = lidar_path +image_name
    com_image= cv2.imread(c)
    hd_image= cv2.imread(hd_img)
    #lidar_image =cv2.imread(ld_img)
        
    #concat_image = cv2.hconcat([sd_image,hd_image])
    #gray1 = cv2.cvtColor(com_image, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(sd_image, cv2.COLOR_BGR2GRAY)
    #gray3 = cv2.cvtColor(lidar_image, cv2.COLOR_BGR2GRAY)


   # img_merged = cv2.merge((gray1,gray2,gray3))
    cv2.imwrite(output_hd_path +str(os.path.basename(c)), hd_image)



