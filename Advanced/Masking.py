import cv2 as cv
import numpy as np

img=cv.imread('cats.jpg')
cv.imshow('Cats',img)

blank=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')

# mask is used to get a specific portion of an image
mask=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)

masked=cv.bitwise_and(img,img,mask=mask)
cv.imshow('masked',masked)

cv.waitKey(0)
