# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:21:45 2020

@author: simon
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Get user supplied values
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv.CascadeClassifier(cascPath)

# Read the first image
image1 = cv.imread("pp1.png")
gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray1,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(image1, (x, y-40), (x+w, y+h+80), (0, 255, 0), 2)
    cut1 =image1[y-40:y+h+80,x:x+w,:] 
    cv.imwrite("cut1.png",cut1)
    
# Read the first image
image2 = cv.imread("pp2.png")
gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray2,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(image2, (x, y-40), (x+w, y+h+80), (0, 255, 0), 2)
    cut2 =image2[y-40:y+h+80,x:x+w,:] 
    cv.imwrite("cut2.png",cut2)
    
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

# Read and resize the two image
dim = (300, 400)
img1 = cv.imread("cut1.png", cv.IMREAD_COLOR)
img2 = cv.imread("cut2.png", cv.IMREAD_COLOR)

img1 = cv.resize(img1, dim, interpolation = cv.INTER_AREA)
img2 = cv.resize(img2, dim, interpolation = cv.INTER_AREA)
length, width, com = img1.shape
plt.figure()
plt.title('cut1')
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.figure()
plt.title('cut2')
plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))

# Apply skin_rgb_threshold to both images to do skin detection
skin1 = skin_rgb_threshold(img1)
plt.figure()
plt.title('Skin Detection on cut1.png')
plt.imshow(cv.cvtColor(skin1, cv.COLOR_BGR2RGB))
skin2 = skin_rgb_threshold(img2)
plt.figure()
plt.title('Skin Detection on cut2.png')
plt.imshow(cv.cvtColor(skin2, cv.COLOR_BGR2RGB))

# Subtract two image and create the mask which has all 
diff = np.zeros((length, width, com), "uint8")
mask = np.zeros((length,width), "uint8")

for i in range(length):
    for j in range(width):
        for k in range(com):
            diff[i, j, k] = abs(int(img1[i, j, k]) - int(img2[i, j, k]))
            if diff[i, j, k] < 40:
#           if diff[i, j, k] < 35:
#           if diff[i, j, k] < 45:
                diff[i, j, k] = 0
     
for i in range(length):
    for j in range(width):
        if diff[i, j, 0] + diff[i, j, 1] + diff[i, j, 2] > 2:
            mask[i, j] = 1
plt.figure()
plt.title('difference')
plt.imshow(cv.cvtColor(diff, cv.COLOR_BGR2GRAY))            
plt.figure()
plt.title('mask')
plt.imshow(mask, cmap='gray')

# Applying opening and closing to the mask
kernel = np.ones((7, 7), np.uint8)
#kernel = np.ones((6, 6), np.uint8)
#kernel = np.ones((8, 8), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
plt.figure()
plt.title('mask after opening and closing (8*8)')
plt.imshow(mask, cmap='gray')

# Applying dilation to the mask to increase the mask size a little bit
#mask = cv.erode(mask, kernel)
mask = cv.dilate(mask, kernel)
mask = cv.dilate(mask, kernel)
plt.figure()
plt.title('mask after dilation')
plt.imshow(mask, cmap='gray')

# Select the pixels in img1 whick on the non-black area in mask,
# then merge them into img2
merge = np.zeros((length, width, com), "uint8")
for i in range(length):
    for j in range(width):
        merge[i, j, :] = img2[i, j, :]
        if mask[i, j] > 0:
            merge[i, j, :] = img1[i, j, :]
plt.figure()
plt.title('merged image')
plt.imshow(cv.cvtColor(merge, cv.COLOR_BGR2RGB)) 

# Try Gaussian filter to smooth the image
G_ker=np.array([
        [ 1,  4,  4,  4,  1],
        [ 4, 16, 26, 16,  4],
        [ 7, 26, 41, 26,  7],
        [ 4, 16, 26, 16,  4],
        [ 1,  4,  4,  4,  1]]
)
G_ker =G_ker/np.sum(G_ker)
gaussian = cv.filter2D(merge, -1, G_ker)
plt.figure()
plt.title('Skin smoothing by Gaussian filter')
plt.imshow(cv.cvtColor(gaussian, cv.COLOR_BGR2RGB))
cv.imwrite("final.png",gaussian)