import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('cats.jpg')
cv.imshow('original',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

blank=np.zeros(img.shape[:2],dtype='uint8')
mask=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

masked=cv.bitwise_and(img,img,mask=mask)
cv.imshow('masked',masked)

# Grayscale Histogram of masked part

gray_hist=cv.calcHist([gray],[0],mask,[256],[0,256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('No. of pixels')
plt.plot(gray_hist)
plt.show()

# Color Histogram of masked part

plt.figure()
plt.title('Coloured Histogram')
plt.xlabel('Bins')
plt.ylabel('No. of pixels')
colors=['b','g','r']
for (i,col) in enumerate(colors):
    color_hist=cv.calcHist([img],[i],mask,[256],[0,256])
    plt.plot(color_hist,color=col)
plt.show()

cv.waitKey(0)


