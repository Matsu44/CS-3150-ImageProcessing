# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:34:34 2020

@author: simon
"""

import numpy as np
import cv2 
import math
from matplotlib import pyplot as plt

# Read image
im = cv2.imread('./image.jpg')
# Change to grayscale
im2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# Remove and change the pixels outside the foreground to be black
length, width = im2.shape
radius = 0.35*min(length,width)
for i in range(length):     # this is the row
    for j in range(width):  # this is the column      
        if math.sqrt((i-length/2)**2 + (j-width/2)**2) > radius:
             im2[i,j] = 0
# Print the new image
plt.figure()
plt.imshow(im2,cmap='gray', vmin=0, vmax=255)  

# Gamma / Power law
im3 = np.zeros((length, width))
imgN = im2 / 255
im3 = imgN ** 0.7
plt.figure()
plt.imshow(im3,cmap='gray', vmin=0, vmax=1)