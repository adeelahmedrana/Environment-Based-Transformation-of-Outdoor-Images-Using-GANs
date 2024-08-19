# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 18:56:52 2022

@author: adeel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 18:30:11 2022

@author: adeel
"""

import glob
import os
import cv2

#orignal_output ="C:/Users/adeel/Python/rain/orignal/test/"
output_gray ="C:/Users/adeel/Python/rain/blur/train/"

#output_rain =r"C:\Users\adeel\Python\rain\rain\"
#os.chdir(output_rain)

input_image = hd_path_output ="C:/Users/adeel/Python/rain/orignal/train/*.jpg"


#output_concat_path = "/home/ahd7rng/DS/Dataset/Output dil 3/"


hd_names = [x for x in glob.glob(input_image)]
#sd_names = [x for x in glob.glob(output)]
hd_names
print(os.path.basename(hd_names[0]))


for h in (hd_names):
    #if os.path.basename(h) == os.path.basename(s):
        #print("same name")
        
    hd_image =cv2.imread(h)
    dst = cv2.GaussianBlur(hd_image,(11,11),cv2.BORDER_DEFAULT)
        
        #concat_image = cv2.hconcat([hd_image,sd_image])
    cv2.imwrite(output_gray +str(os.path.basename(h)), dst)
    #cv2.imwrite(output_rain +str(os.path.basename(h)), right_half)
    print('done')
    
print('left loop')
   # print('The Dataset Size is:'  + len(hd_names))
    #print('done')
    