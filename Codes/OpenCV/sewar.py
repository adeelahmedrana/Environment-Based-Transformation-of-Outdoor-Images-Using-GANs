# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:27:57 2022

@author: adeel
"""

import cv2
#import numpy as np
#from sewar import full_ref
from skimage import measure, metrics
from sewar.full_ref import uqi,mse,rmse,ssim

import sewar
import glob
import os

import csv

pix2pix_image ="C:/Users/adeel/Python/Outputs/p2p/rain/*.jpg" # Gray

orignal = "C:/Users/adeel/Python/Outputs/p2p/orignal/rain/*.jpg" #Gray

s_image = [x for x in glob.glob(orignal)]
s_image

p_image = [x for x in glob.glob(pix2pix_image)]
p_image


###########################################################################
# Mean Sqaured Error MSE

with open('C:/Users/adeel/Python/comparison/Orignal_with_p2p/mse.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    

    for p,s in zip(p_image,s_image):
        
        main =cv2.imread(s)
        ref =cv2.imread(p)
        
        #mse_skimg = metrics.mean_squared_error(main, ref)
        a =mse(main,ref)
        #row = [mse_skimg]
        row=[a]
        print(row)
        writer.writerow([row])
        
        
        print('done with MSE: ')
        
##############################################################################
###########################################################################
# Structural Similarity Index (SSIM)

with open('C:/Users/adeel/Python/comparison/Orignal_with_p2p/ssim.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    

    for p,s in zip(p_image,s_image):
        
        main =cv2.imread(s)
        ref =cv2.imread(p)
        
        #mse_skimg = metrics.mean_squared_error(main, ref)
        a =ssim(main,ref)
        #row = [mse_skimg]
        row=[a]
        print(row)
        writer.writerow([row])
        
        
        print('done with SSIM: ')
        
##############################################################################
###########################################################################
#  Universal Quality Image Index (UQI)

with open('C:/Users/adeel/Python/comparison/Orignal_with_p2p/uqi.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    

    for p,s in zip(p_image,s_image):
        
        main =cv2.imread(s)
        ref =cv2.imread(p)
        
        #mse_skimg = metrics.mean_squared_error(main, ref)
        a =uqi(main,ref)
        #row = [mse_skimg]
        row=[a]
        print(row)
        writer.writerow([row])
        
        
        print('done with UQI: ')
        
##############################################################################
# Root Mean Sqaured Error (RMSE)

with open('C:/Users/adeel/Python/comparison/Orignal_with_p2p/rmse.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    

    for p,s in zip(p_image,s_image):
        
        main =cv2.imread(s)
        ref =cv2.imread(p)
        
        #mse_skimg = metrics.mean_squared_error(main, ref)
        a =rmse(main,ref)
        #row = [mse_skimg]
        row=[a]
        print(row)
        writer.writerow([row])
        
        
        print('done with RMSE: ')
        
##############################################################################


# Self Checkup

###########################################################################
# Root Mean Sqaured Error (RMSE)

with open('C:/Users/adeel/Python/comparison/Orignal_with_p2p/self_rmse.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    

    for p,s in zip(p_image,s_image):
        
        main =cv2.imread(s)
        ref =cv2.imread(p)
        
        #mse_skimg = metrics.mean_squared_error(main, ref)
        a =rmse(main,main)
        #row = [mse_skimg]
        row=[a]
        print(row)
        writer.writerow([row])
        
        
        print('done with self_RMSE: ')
        
##############################################################################

###########################################################################
# Mean Sqaured Error MSE

with open('C:/Users/adeel/Python/comparison/Orignal_with_p2p/self_mse.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    

    for p,s in zip(p_image,s_image):
        
        main =cv2.imread(s)
        ref =cv2.imread(p)
        
        #mse_skimg = metrics.mean_squared_error(main, ref)
        a =mse(main,main)
        #row = [mse_skimg]
        row=[a]
        print(row)
        writer.writerow([row])
        
        
        print('done with self_MSE: ')
        
##############################################################################
###########################################################################
# Structural Similarity Index (SSIM)

with open('C:/Users/adeel/Python/comparison/Orignal_with_p2p/self_ssim.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    

    for p,s in zip(p_image,s_image):
        
        main =cv2.imread(s)
        ref =cv2.imread(p)
        
        #mse_skimg = metrics.mean_squared_error(main, ref)
        a =ssim(main,main)
        #row = [mse_skimg]
        row=[a]
        print(row)
        writer.writerow([row])
        
        
        print('done with self_SSIM: ')
        
##############################################################################
###########################################################################
#  Universal Quality Image Index (UQI)

with open('C:/Users/adeel/Python/comparison/Orignal_with_p2p/self_uqi.csv','w') as f1:
    writer=csv.writer(f1, delimiter='\t',lineterminator='\n',)
    

    for p,s in zip(p_image,s_image):
        
        main =cv2.imread(s)
        ref =cv2.imread(p)
        
        #mse_skimg = metrics.mean_squared_error(main, ref)
        a =uqi(main,main)
        #row = [mse_skimg]
        row=[a]
        print(row)
        writer.writerow([row])
        
        
        print('done with self_UQI: ')

###########################################################################
    
print('left loop')
   # print('The Dataset Size is:'  + len(orignal))
    #print('done')