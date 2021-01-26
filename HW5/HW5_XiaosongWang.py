import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def MSE(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
     
    return mse

def make_g_noise(length, width, height, variance):
    sigmas = math.sqrt(variance)
    noise = sigmas * np.random.randn(length, width, height)
    return noise

##############################################################################
#                           Main Program                                     #
##############################################################################
imA =cv.imread("./Saliency0.jpg")
# imA =cv.imread("./Saliency1.jpeg")
# imA =cv.imread("./Saliency2.jpg")
# imA =cv.imread("./Saliency3.png")
# imA =cv.imread("./Saliency4.png")
imA = cv.cvtColor(imA, cv.COLOR_BGR2RGB)
length = imA.shape[0]
width = imA.shape[1]
height = imA.shape[2]

plt.figure()
plt.imshow(imA)
plt.title("Original")

g_noise = make_g_noise(length, width, height, 50000)
imB = imA + g_noise

mseV= MSE(imA,imB)

plt.figure()
plt.imshow(imB)
plt.title("Gaussian: MSE: %.4f" % (mseV))


saliency = cv.saliency.StaticSaliencySpectralResidual_create()
# saliency map values are between 0-1
success, sal_map = saliency.computeSaliency(imA)

plt.figure()
plt.imshow(sal_map,cmap='gray')
plt.title("saliency")

sal_map = sal_map * 255

for i in range(length):
    for j in range(width):
        for k in range(height):
            if sal_map[i,j] <= 30:
                imA[i,j,k] = 0
                imB[i,j,k] = 0

mseV_2= MSE(imA,imB)
print("mse value:" + str(mseV_2))
 