import cv2 as cv
import numpy as np

img=cv.imread('cats.jpg')
cv.imshow('original',img)

blank=np.zeros(img.shape[:2],dtype='uint8')

# Splitting into 3 color channels
b,g,r=cv.split(img)

blue=cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])

cv.imshow('blue',blue)
cv.imshow('green',green)
cv.imshow('red',red)

print(img.shape)
print(b.shape)

cv.waitKey(0)
