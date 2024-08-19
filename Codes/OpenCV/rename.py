# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:46:07 2022

@author: adeel
"""

import os
os.getcwd()
collection = "C:/Users/adeel/Python/Outputs/star/gray"
for i, filename in enumerate(os.listdir(collection)):
    os.rename("C:/Users/adeel/Python/Outputs/star/gray/" + filename, "C:/Users/adeel/Python/Outputs/star/gray/" + str(i) + ".jpg")