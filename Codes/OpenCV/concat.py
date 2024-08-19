# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:03:11 2022

@author: adeel
"""

import glob
import os
import cv2


orignal_output ="C:/Users/adeel/Python/rain/orignal/test/*.jpg"

input_x ="C:/Users/adeel/Python/rain/gray/test/*.jpg"

output_concat_path = "C:/Users/adeel/Python/rain/gray_input/test/"

hd_names = [x for x in glob.glob(orignal_output)]
sd_names = [x for x in glob.glob(input_x)]



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
        