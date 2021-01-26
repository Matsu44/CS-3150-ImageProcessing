# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:36:58 2020

@author: simon

@referance: https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio
            https://stackoverflow.com/questions/55178229/importerror-cannot-import-name-structural-similarity-error
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage import measure
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

##############################################################################
# Read image
imA =cv.imread("./lena_g.bmp")
imA = cv.cvtColor(imA, cv.COLOR_BGR2GRAY)
length = imA.shape[0]
width = imA.shape[1]

##############################################################################
# Make Gaussian noise function
def make_g_noise( height, width , variance):
    sigmas = math.sqrt(variance)
    noise = sigmas * np.random.randn(height,width)
    return noise

##############################################################################
# Create different image

# Add Gaussian noise
g_noise = make_g_noise(length, width, 150)
im_g_noisy = imA + g_noise

# Add Uniform noise
u_noise = np.random.uniform(-25, 25, (length, width))
im_u_noisy = imA + u_noise

# Applying Histogram Equalization
im_equ = cv.equalizeHist(imA)

# Applying Contrast Stretching
im2=np.zeros((length, width))
mx=np.amax(imA)
mn=np.amin(imA)
cs=(imA-mn)/(mx-mn)*255

# Applying median filter to the uniform noisy image
height,width = np.shape(im_u_noisy)
median = np.zeros((height,width),dtype=float)
for i in range(1,height-2):
    for j in range(1,width-2):
        sorted_pixels = sorted(np.ndarray.flatten(im_u_noisy[i-1:i+2,j-1:j+2]))
        median[i][j] = sorted_pixels[4]
        
##############################################################################
# Image Quantity Assessment

# Origin     
mseV = mse(imA, imA)
psnrV = psnr(imA, imA)
ssimV = measure.compare_ssim(imA, imA)
plt.figure()
plt.title("Origin: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
plt.imshow(imA,cmap='gray')

# Gaussian noise
mseV = mse(imA, im_g_noisy)
psnrV = psnr(imA, im_g_noisy)
ssimV = measure.compare_ssim(imA, im_g_noisy)
plt.figure()
plt.title("Guassian: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
plt.imshow(im_g_noisy,cmap='gray')
cv.imwrite("Guassian.jpg", im_g_noisy)

# Uniform noise
mseV = mse(imA, im_u_noisy)
psnrV = psnr(imA, im_u_noisy)
ssimV = measure.compare_ssim(imA, im_u_noisy)
plt.figure()
plt.title("Uniform: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
plt.imshow(im_u_noisy,cmap='gray')
cv.imwrite("Uniform.jpg", im_u_noisy)

# Histogram Equalization
mseV = mse(imA, im_equ)
psnrV = psnr(imA, im_equ)
ssimV = measure.compare_ssim(imA, im_equ)
plt.figure()
plt.title("Histogram Equalization: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
plt.imshow(im_equ,cmap='gray')
cv.imwrite("Histogram Equalization.jpg", im_equ)

# Contrast Stretching
mseV = mse(imA, cs)
psnrV = psnr(imA, cs)
ssimV = measure.compare_ssim(imA, cs)
plt.figure()
plt.title("Contrast Stretching: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
plt.imshow(cs,cmap='gray')
cv.imwrite("Contrast Stretching.jpg", cs)

# Median filter after Uniform noise
mseV = mse(imA, median)
psnrV = psnr(imA, median)
ssimV = measure.compare_ssim(imA, median)
plt.figure()
plt.title("Median: MSE: %.4f, PSNR: %.4f, SSIM: %.4f" % (mseV, psnrV, ssimV))
plt.imshow(median,cmap='gray')
cv.imwrite("Median.jpg", median)


