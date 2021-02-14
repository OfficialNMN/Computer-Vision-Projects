import cv2 as cv
import numpy as np

img=cv.imread(r'C:\Users\NAMANJEET SINGH\PycharmProjects\OpenCVProject\pics\park.jpg')
cv.imshow('park',img)

grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('grey',grey)

# Laplacian Edge detection
lap=cv.Laplacian(grey,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow('Laplacian',lap)

# Sobel Edge Detection
sobelx=cv.Sobel(grey,cv.CV_64F,1,0)
sobely=cv.Sobel(grey,cv.CV_64F,0,1)
combined_sobel=cv.bitwise_or(sobelx,sobely)
cv.imshow('Combined Sobel',combined_sobel)

# Canny Edge Detection
canny=cv.Canny(grey,150,175)
cv.imshow('Canny',canny)

cv.waitKey(0)





