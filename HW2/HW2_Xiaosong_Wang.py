# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:54:30 2020

@author: simon
"""

import numpy as np
import cv2 
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

# Read image
img=cv2.imread('./image.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.title('original image')
plt.imshow(img_gray,cmap='gray', vmin=0, vmax=255) 

# ****************************************************************************
# Average filter
# ****************************************************************************

avg_kern_1 = np.array(
    [[1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]]
)
avg_kern_2 = np.array(
    [[1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36],
    [1/36,1/36,1/36,1/36,1/36,1/36]]
)

# 3*3 Average filter
average_1 = cv2.filter2D(img_gray,-1,avg_kern_1)    
plt.figure()
plt.title('average filter(3*3)')
plt.imshow(average_1,cmap='gray', vmin=0, vmax=255)

# 5*5 Average filter 
average_2 = cv2.filter2D(img_gray,-1,avg_kern_2)    
plt.figure()
plt.title('average filter(5*5)')
plt.imshow(average_2,cmap='gray', vmin=0, vmax=255)  

# ****************************************************************************
# Sobel filter
# ****************************************************************************

sobel_vert = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]]
)
sobel_horiz = sobel_vert.T

# use OpenCV functions
dst_vert = cv2.filter2D(img_gray, -1, sobel_vert) 
dst_horiz = cv2.filter2D(img_gray, -1, sobel_horiz) 
plt.figure()
plt.title('Vert_by opencv')
plt.imshow(dst_vert,cmap='gray', vmin=0, vmax=255) 
plt.figure()
plt.title('Horiz_ by opencv')
plt.imshow(dst_horiz,cmap='gray', vmin=0, vmax=255) 

# Try cv.filter2D to get the gradient
grad = np.sqrt(np.square(dst_vert) + np.square(dst_horiz)).astype(int)
plt.figure()
plt.title('Gradient Edge by opencv')
plt.imshow(grad,cmap='gray', vmin=0, vmax=255)

# Max Edge by opencv
grad2=np.maximum(dst_vert, dst_horiz) 
plt.figure()
plt.title('Max Edge by opencv')
plt.imshow(grad2,cmap='gray', vmin=0, vmax=255)

# Try convolve2d
dst_vert = convolve2d(img_gray,sobel_vert, mode='same', boundary = 'symm', fillvalue=0)
dst_horiz = convolve2d(img_gray,sobel_horiz, mode='same', boundary = 'symm', fillvalue=0)
grad3 = np.sqrt(np.square(dst_vert) + np.square(dst_horiz))
grad3 *= 255.0 / np.max(grad3)
plt.figure()
plt.title('Gradient Edge by convolve2d')
plt.imshow(grad3,cmap='gray', vmin=0, vmax=255)

# ****************************************************************************
# Laplacian filter
# ****************************************************************************

l_kern_1 = np.array([
  [ 0.0,  1.0, 0.0],
  [ 1.0, -4.0, 1.0],
  [ 0.0,  1.0, 0.0]]
)
l_kern_2 = np.array([
  [1.0,  1.0, 1.0],
  [1.0, -8.0, 1.0],
  [1.0,  1.0, 1.0]]
    )

Edge_l = cv2.filter2D(img_gray, -1, l_kern_1)
Edge_2 = cv2.filter2D(img_gray, -1, l_kern_2) 

# Laplacian Edge 1
plt.figure()
plt.title('Laplacian Edge')
plt.imshow(Edge_l,cmap='gray', vmin=0, vmax=255)

# Laplacian Edge 2
plt.figure()
plt.title('Laplacian Edge')
plt.imshow(Edge_2,cmap='gray', vmin=0, vmax=255)


# ****************************************************************************
# Median filter
# ****************************************************************************

height,width = np.shape(img_gray)
median = np.zeros((height,width),dtype=float)
for i in range(1,height-2):
    for j in range(1,width-2):
        sorted_pixels = sorted(np.ndarray.flatten(img_gray[i-1:i+2,j-1:j+2]))
        median[i][j] = sorted_pixels[4]
plt.figure()
plt.title('Median Filter')
plt.imshow(median,cmap='gray', vmin=0, vmax=255) 

# ****************************************************************************
# Guassian filter
# ****************************************************************************

G_ker=np.array([
        [ 1,  4,  4,  4,  1],
        [ 4, 16, 26, 16,  4],
        [ 7, 26, 41, 26,  7],
        [ 4, 16, 26, 16,  4],
        [ 1,  4,  4,  4,  1]]
)
G_ker =G_ker/np.sum(G_ker)
Gaussian = cv2.filter2D(img_gray, -1, G_ker) 
plt.figure()
plt.title('Gaussian Blur')
plt.imshow(Gaussian,cmap='gray', vmin=0, vmax=255) 

# ****************************************************************************
# Maximum filter
# https://www.geeksforgeeks.org/spatial-filtering-and-its-types/#:~:text=Spatial%20Filtering%20technique%20is%20used,mask%20traverses%20all%20image%20pixels.
# ****************************************************************************

height,width = np.shape(img_gray)
max = np.zeros((height,width),dtype=float)
for i in range(1,height-2):
    for j in range(1,width-2):
        sorted_pixels = sorted(np.ndarray.flatten(img_gray[i-1:i+2,j-1:j+2]))
        max[i][j] = sorted_pixels[8]
plt.figure()
plt.title('Maximum Filter filter')
plt.imshow(max,cmap='gray', vmin=0, vmax=255) 


