# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:53:12 2020

@author: simon
"""

import numpy as np
import cv2 
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import math

# Read image
img=cv2.imread('./iris.bmp')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.title('original image')
plt.imshow(img_gray,cmap='gray', vmin=0, vmax=255) 

# Applying average filter to generate a smooth version
avg_kern = np.array(
    [[1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36]]
)

average = cv2.filter2D(img_gray,-1,avg_kern)    
plt.figure()
plt.title('average filter')
plt.imshow(average,cmap='gray', vmin=0, vmax=255)  

# Applying sobel filter to find obvious edges
sobel_vert = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]]
)
sobel_horiz = sobel_vert.T

dst_vert = convolve2d(average, sobel_vert, mode='same', boundary = 'symm', fillvalue=0)
dst_horiz = convolve2d(average, sobel_horiz, mode='same', boundary = 'symm', fillvalue=0)
edge = np.sqrt(np.square(dst_vert) + np.square(dst_horiz))
edge *= 255.0 / np.max(edge)
plt.figure()
plt.title('edge detected by sobel filter')
plt.imshow(edge,cmap='gray', vmin=0, vmax=255)

# Create disc kernel
disc_kern = np.zeros((100,100), dtype = float)
for i in range(100):
    for j in range(100):
        dist = math.sqrt(math.pow(i-50,2) + math.pow(j-50,2))
        if dist > 35 and dist < 45:
            disc_kern[i,j] = 1
            
# Applying disc kernel to detect the pupil boundary
disc = convolve2d(edge, disc_kern, mode='same', boundary = 'symm', fillvalue=0)
disc = np.absolute(disc)
disc *= 255.0 / np.max(disc)
plt.figure()
plt.title('detected pupil boundary')
plt.imshow(disc,cmap='gray', vmin=0, vmax=255)

# Find the brightest pixel and erase it, and black out everything outside
def in_circle(x, y, center_x, center_y, radius):
    distance = math.sqrt(math.pow(x-center_x,2) + math.pow(y-center_y,2))
    return (distance < radius)

max_pixel_value = 0
max_x = 0
max_y = 0
w, h = np.shape(disc)
for i in range(w):
    for j in range(h):
        if disc[i][j] > max_pixel_value:
            max_x = i
            max_y = j
            max_pixel_value = disc[i][j]
            
for i in range(w):
    for j in range(h):
        if(not in_circle(i, j, max_x, max_y, 45)):
            img[i][j] = 0
            
# Plot the image
plt.figure()
plt.title('final result')
plt.imshow(img,cmap='gray', vmin=0, vmax=255)
cv2.imwrite("final_result.jpg", img)


