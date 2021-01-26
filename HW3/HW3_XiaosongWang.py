# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:47:58 2020

@author: simon
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# rgb threshold from the paper
def skin_rgb_threshold(src):

    b = src[:, :, 0].astype(np.int16)
    g = src[:, :, 1].astype(np.int16)
    r = src[:, :, 2].astype(np.int16)

    skin_mask =                                    \
        (r > 96) & (g > 40) & (b > 10)             \
        & ((src.max() - src.min()) > 15)           \
        & (np.abs(r-g) > 15) & (r > g) & (r > b)    

    return src * skin_mask.reshape(skin_mask.shape[0], skin_mask.shape[1], 1)

##############################################################################
# Create a filter to find the peak of the histogram
def find_local_min(hist):

    kern = np.array(
            [2,0,0,0,
             2,0,0,0,
             2,0,0,0,
             2,0,0,0,
             1,0,0,0,
             1,0,0,0,
             1,0,0,0,
             1,0,0,0,
             -3,-3,-3,-3
             -3,-3,-3,-3
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,1
             ,0,0,0,2
             ,0,0,0,2
             ,0,0,0,2
             ,0,0,0,2])

    hist[0] = 0
    deriv = np.convolve(hist, kern, mode='same')
    threshold = deriv.argmax()
    return threshold, deriv

##############################################################################
# Read face_good image
img1 = cv.imread("face_good.bmp", cv.IMREAD_COLOR)
plt.figure()
plt.title('Good Face Original Image in RGB')
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))

# Call the skin_rgb_threshold to detect the skin and show figure with skin only
skin1 = skin_rgb_threshold(img1)
plt.figure()
plt.title('Skin Detection on Good Face')
plt.imshow(cv.cvtColor(skin1, cv.COLOR_BGR2RGB))
cv.imwrite("face_good_skin_only.jpg", skin1)

##############################################################################
# Read face_dark image
img2 = cv.imread("face_dark.bmp", cv.IMREAD_COLOR)
plt.figure()
plt.title('Bad Face Original Image in RGB')
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))

# Convert to LUV
luv = cv.cvtColor(img2, cv.COLOR_BGR2LUV)
l = luv[:,:,0]

# Generate the histogram of luminance component of the LUV image
hist, bins = np.histogram(l.ravel(), 256, [0, 256])

# Apply the 1D filter to the histogram to find the threshold
threshold, deriv = find_local_min(hist)

# Set all values < threshold to 0 and replace the new L information to the LUV image
l = (l < threshold) * l
luv[:,:,0] = l

# Convert the new LUV back to RGB then call the skin_rgb_threshold to detect the skin and show figure with skin only
new = cv.cvtColor(luv, cv.COLOR_LUV2BGR)
skin2 = skin_rgb_threshold(new)
plt.figure()
plt.title('Skin Detection on Dark Face')
plt.imshow(cv.cvtColor(skin2, cv.COLOR_BGR2RGB))
cv.imwrite("face_good_skin_only(dark).jpg", skin2)