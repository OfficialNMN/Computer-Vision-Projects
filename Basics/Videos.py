import cv2
import datetime

capture=cv2.VideoCapture('dog.mp4')

while True:
    isTrue,frame=capture.read()
    cv2.imshow('Video',frame)

    if cv2.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv2.destroyAllWindows()
