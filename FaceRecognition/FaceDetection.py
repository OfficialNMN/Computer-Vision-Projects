import cv2 as cv

img=cv.imread(r'C:\Users\NAMANJEET SINGH\PycharmProjects\OpenCVProject\pics\group.jpg')

grey=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

haar_cascade=cv.CascadeClassifier('haar_face.xml')

faces=haar_cascade.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=6)

print(f'{len(faces)} found')

for x,y,w,h in faces:
    cv.rectangle(img,(x,y),(x+h,y+w),(0,255,0),2)

cv.imshow('DetectedFaces',img)
cv.waitKey(0)