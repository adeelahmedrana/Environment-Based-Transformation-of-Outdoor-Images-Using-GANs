# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:16:05 2022

@author: adeel
"""

import glob
import os
import cv2

#orignal_output ="C:/Users/adeel/Python/Star/a/StarGAN_simple/dataset/restoration/train/domain1/"
orignal_output ="C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/SEASONS/resized/summer/"

#output_rain =r"C:\Users\adeel\Python\rain\rain\"
#os.chdir(output_rain)

input_image = hd_path_output ="C:/Users/adeel/OneDrive/Desktop/TUI/Project/Datasets/SEASONS/train/summer/*.png"


#output_concat_path = "/home/ahd7rng/DS/Dataset/Output dil 3/"


hd_names = [x for x in glob.glob(input_image)]
#sd_names = [x for x in glob.glob(output)]
hd_names
print(os.path.basename(hd_names[0]))


for h in (hd_names):
    #if os.path.basename(h) == os.path.basename(s):
        #print("same name")
        
    hd_image =cv2.imread(h) #369 by star
    dim = (256, 256)
  
# resize image
    image_num = 1
    resized = cv2.resize(hd_image, dim, interpolation = cv2.INTER_AREA)
        
        #concat_image = cv2.hconcat([hd_image,sd_image])
    cv2.imwrite(orignal_output +str('.jpg' , resized)
    image_num = image_num + 1
    #cv2.imwrite(output_rain +str(os.path.basename(h)), right_half)
    print('done' + str(image_num))
     
    
print('left loop')
   # print('The Dataset Size is:'  + len(hd_names))
    #print('done')
    
        
    